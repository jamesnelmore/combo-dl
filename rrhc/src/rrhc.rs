use std::sync::atomic::{AtomicUsize, Ordering};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::group::Group;

/// The parameters of the directed strongly regular graph we are searching for:
/// a connection set of size `k` in a group of order `n` satisfying the DSRG
/// equation `A^2 = t*I + lambda*A + mu*(J - I - A)`.
pub struct DSRGParameterSet {
    pub n: usize,
    pub k: usize,
    pub t: usize,
    pub lambda: usize,
    pub mu: usize,
}

/// A candidate connection set together with its objective error. An error of
/// zero means the set solves the DSRG equation.
#[derive(Debug, Clone)]
pub struct ScoredSet {
    pub connection_set: Vec<usize>,
    pub error: i64,
}

/// Random-restart hill climbing, parallelized across cores with rayon: each
/// restart climbs from a fresh random subset on its own worker thread, and the
/// winning set is the one with the lowest error, ties broken by lowest restart
/// index. Each restart derives an independent RNG from `base_seed` and its index.
///
/// The result is fully deterministic for a fixed `base_seed`, independent of
/// core count and thread scheduling: the winner is a pure function of the
/// per-restart outcomes. Once some restart solves exactly at index `j`, restarts
/// at index `>= j` cannot win (a lower or equal index already ties or beats
/// them), so they short-circuit -- but every index below `j` still runs, so the
/// winner never depends on who finished first.
/// `progress` is incremented once per restart index processed (including ones
/// short-circuited by the early exit), so a caller can report live progress
/// across a whole sweep from one shared counter.
pub fn random_restart_hill_climb(
    group: &Group,
    params: &DSRGParameterSet,
    max_restarts: usize,
    base_seed: u64,
    progress: &AtomicUsize,
) -> ScoredSet {
    // Lowest restart index known to have solved exactly (usize::MAX = none yet).
    let min_solved = AtomicUsize::new(usize::MAX);
    (0..max_restarts)
        .into_par_iter()
        .filter_map(|restart| {
            progress.fetch_add(1, Ordering::Relaxed);
            if restart >= min_solved.load(Ordering::Relaxed) {
                return None; // a lower/equal index already solves; this can't win
            }
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(restart as u64));
            let start = random_subset(group, params.k, &mut rng);
            let result = hill_climb(group, params, start);
            if result.error == 0 {
                min_solved.fetch_min(restart, Ordering::Relaxed);
            }
            Some((restart, result))
        })
        .reduce_with(|a, b| if (b.1.error, b.0) < (a.1.error, a.0) { b } else { a })
        .map(|(_, best)| best)
        .expect("max_restarts must be at least 1")
}

/// The DSRG objective ("error"): the sum of squared coefficients of the
/// residual group-ring element `A^2 - t*I - lambda*A - mu*(J - I - A)`, where
/// `A` is the indicator of `connection_set`. Returns 0 exactly when the set
/// solves the DSRG equation. Kept as the simple, O(k^2) reference definition;
/// hill climbing uses the incremental [`Objective`] below. Only used as the
/// test oracle that validates the incremental scorer.
#[cfg(test)]
fn objective_error(group: &Group, params: &DSRGParameterSet, connection_set: &[usize]) -> i64 {
    let mut in_set = vec![false; group.n];
    for &element in connection_set {
        in_set[element] = true;
    }

    // A^2 in the group ring: for each ordered pair (a, b) drawn from the set,
    // the product a*b contributes 1 to its coefficient.
    let mut square = vec![0i64; group.n];
    for &a in connection_set {
        for &b in connection_set {
            square[group.mul(a, b)] += 1;
        }
    }

    let t = params.t as i64;
    let lambda = params.lambda as i64;
    let mu = params.mu as i64;

    let mut total = 0i64;
    for element in 0..group.n {
        let a_coeff = in_set[element] as i64; // coefficient of A
        let i_coeff = (element == group.identity) as i64; // coefficient of I
        let rest_coeff = 1 - a_coeff - i_coeff; // coefficient of (J - I - A)
        let residual = square[element] - t * i_coeff - lambda * a_coeff - mu * rest_coeff;
        total += residual * residual;
    }
    total
}

/// Incremental DSRG objective for a single candidate. It maintains the `A^2`
/// coefficients, the residual coefficients, and the total error, so that scoring
/// a one-element swap costs O(k) rather than the O(k^2 + n) of a full recompute.
///
/// A swap `remove g, add h` changes `A` by `d = e_h - e_g`, so
/// `A'^2 = A^2 + A*d + d*A + d^2` differs in only O(k) coefficients (the group
/// products a*h, a*g, h*a, g*a for members a, plus four from d^2). Only those
/// coefficients, and g/h themselves (whose membership changes), affect the error.
struct Objective<'a> {
    group: &'a Group,
    t: i64,
    lambda: i64,
    mu: i64,
    members: Vec<usize>, // current connection set
    in_set: Vec<bool>,   // membership indicator, indexed by element
    square: Vec<i64>,    // A^2 coefficients
    residual: Vec<i64>,  // residual coefficients
    error: i64,          // sum of squared residual coefficients
    // Scratch reused across swap evaluations: `scratch[idx]` holds the pending
    // delta to square[idx] for the indices listed in `touched` (0 elsewhere).
    scratch: Vec<i64>,
    touched: Vec<usize>,
    marked: Vec<bool>,
}

impl<'a> Objective<'a> {
    fn new(group: &'a Group, params: &DSRGParameterSet, connection_set: &[usize]) -> Self {
        debug_assert_eq!(group.n, params.n, "group order must match the parameter set");
        let n = group.n;
        let mut in_set = vec![false; n];
        for &element in connection_set {
            in_set[element] = true;
        }
        let mut square = vec![0i64; n];
        for &a in connection_set {
            for &b in connection_set {
                square[group.mul(a, b)] += 1;
            }
        }

        let mut obj = Objective {
            group,
            t: params.t as i64,
            lambda: params.lambda as i64,
            mu: params.mu as i64,
            members: connection_set.to_vec(),
            in_set,
            square,
            residual: vec![0; n],
            error: 0,
            scratch: vec![0; n],
            touched: Vec::new(),
            marked: vec![false; n],
        };
        for element in 0..n {
            let r = obj.residual_at(element, obj.square[element], obj.in_set[element]);
            obj.residual[element] = r;
            obj.error += r * r;
        }
        obj
    }

    /// Residual coefficient at `element` given its `A^2` coefficient and whether
    /// it is currently a member of the connection set.
    #[inline]
    fn residual_at(&self, element: usize, square: i64, member: bool) -> i64 {
        let a = member as i64;
        let i = (element == self.group.identity) as i64;
        square - self.t * i - self.lambda * a - self.mu * (1 - a - i)
    }

    #[inline]
    fn touch(&mut self, idx: usize) {
        if !self.marked[idx] {
            self.marked[idx] = true;
            self.touched.push(idx);
        }
    }

    /// Accumulate into `scratch` / `touched` the `A^2` deltas of the move that
    /// removes every element of `removed` and adds every element of `added`, and
    /// mark those elements (their membership changes). Does not touch
    /// `square`/`residual`/`error`.
    ///
    /// Writing `d = sum(added) - sum(removed)` for the change to the indicator
    /// `A`, the square changes by `A*d + d*A + d^2`, which touches only O(k *
    /// |delta|) coefficients. `removed` must be current members and `added`
    /// current non-members (the DSRG search passes singletons, the inverse-closed
    /// SRG search passes whole inverse-pairs).
    fn stage_delta(&mut self, removed: &[usize], added: &[usize]) {
        // A*d + d*A over the current members (A is the pre-move indicator).
        for i in 0..self.members.len() {
            let a = self.members[i];
            for &h in added {
                let ah = self.group.mul(a, h);
                self.scratch[ah] += 1;
                self.touch(ah);
                let ha = self.group.mul(h, a);
                self.scratch[ha] += 1;
                self.touch(ha);
            }
            for &g in removed {
                let ag = self.group.mul(a, g);
                self.scratch[ag] -= 1;
                self.touch(ag);
                let ga = self.group.mul(g, a);
                self.scratch[ga] -= 1;
                self.touch(ga);
            }
        }
        // d^2 = (sum(added) - sum(removed))^2, expanded over element pairs:
        // (+) added*added and removed*removed, (-) the added*removed cross terms.
        for &x in added {
            for &y in added {
                let idx = self.group.mul(x, y);
                self.scratch[idx] += 1;
                self.touch(idx);
            }
            for &y in removed {
                let idx = self.group.mul(x, y);
                self.scratch[idx] -= 1;
                self.touch(idx);
            }
        }
        for &x in removed {
            for &y in added {
                let idx = self.group.mul(x, y);
                self.scratch[idx] -= 1;
                self.touch(idx);
            }
            for &y in removed {
                let idx = self.group.mul(x, y);
                self.scratch[idx] += 1;
                self.touch(idx);
            }
        }
        for &g in removed {
            self.touch(g);
        }
        for &h in added {
            self.touch(h);
        }
    }

    /// Clear the staged deltas, leaving `scratch` all-zero and `touched` empty.
    fn clear_staged(&mut self) {
        for i in 0..self.touched.len() {
            let idx = self.touched[i];
            self.scratch[idx] = 0;
            self.marked[idx] = false;
        }
        self.touched.clear();
    }

    /// Total error of the neighbor that removes `removed` and adds `added`,
    /// without committing.
    fn swap_error_delta(&mut self, removed: &[usize], added: &[usize]) -> i64 {
        self.stage_delta(removed, added);
        let mut error = self.error;
        for i in 0..self.touched.len() {
            let idx = self.touched[i];
            // Membership after the move: `removed` leave, `added` join.
            let member = if removed.contains(&idx) {
                false
            } else if added.contains(&idx) {
                true
            } else {
                self.in_set[idx]
            };
            let new_res = self.residual_at(idx, self.square[idx] + self.scratch[idx], member);
            let old_res = self.residual[idx];
            error += new_res * new_res - old_res * old_res;
        }
        self.clear_staged();
        error
    }

    /// Apply the move `remove `removed`, add `added`` permanently in
    /// O(k * |delta|).
    fn commit_delta(&mut self, removed: &[usize], added: &[usize]) {
        self.stage_delta(removed, added);
        for i in 0..self.touched.len() {
            let idx = self.touched[i];
            self.square[idx] += self.scratch[idx];
        }
        // Update membership before recomputing residuals so they see the new set.
        for &g in removed {
            self.in_set[g] = false;
        }
        for &h in added {
            self.in_set[h] = true;
        }
        self.members.retain(|x| !removed.contains(x));
        self.members.extend_from_slice(added);
        for i in 0..self.touched.len() {
            let idx = self.touched[i];
            let new_res = self.residual_at(idx, self.square[idx], self.in_set[idx]);
            let old_res = self.residual[idx];
            self.error += new_res * new_res - old_res * old_res;
            self.residual[idx] = new_res;
        }
        self.clear_staged();
    }

    /// Single-element swap `remove g, add h` (the DSRG neighborhood): the
    /// special case of [`Objective::swap_error_delta`] with singletons.
    #[inline]
    fn swap_error(&mut self, g: usize, h: usize) -> i64 {
        self.swap_error_delta(&[g], &[h])
    }

    /// Single-element commit `remove g, add h` (the DSRG neighborhood).
    #[inline]
    fn commit(&mut self, g: usize, h: usize) {
        self.commit_delta(&[g], &[h]);
    }
}

/// Best-improvement hill climbing from a starting candidate. Each sweep tries
/// every single-element swap (drop one member, add one non-member that is not
/// the identity), scores each neighbor, and moves to the best strict improver.
/// Stops at a local optimum, or early if it reaches error 0.
fn hill_climb(group: &Group, params: &DSRGParameterSet, start: Vec<usize>) -> ScoredSet {
    let mut obj = Objective::new(group, params, &start);
    if obj.error == 0 {
        return ScoredSet {
            connection_set: obj.members,
            error: 0,
        };
    }

    loop {
        let mut best_error = obj.error;
        let mut best_move: Option<(usize, usize)> = None;

        for slot in 0..obj.members.len() {
            let removed = obj.members[slot];
            for added in 0..group.n {
                if added == group.identity || obj.in_set[added] {
                    continue;
                }
                let error = obj.swap_error(removed, added);
                if error == 0 {
                    obj.commit(removed, added);
                    return ScoredSet {
                        connection_set: obj.members,
                        error: 0,
                    };
                }
                if error < best_error {
                    best_error = error;
                    best_move = Some((removed, added));
                }
            }
        }

        match best_move {
            Some((g, h)) => obj.commit(g, h),
            None => {
                return ScoredSet {
                    connection_set: obj.members,
                    error: obj.error,
                };
            }
        }
    }
}

/// Draw `size` distinct non-identity elements by rejection sampling.
fn random_subset(group: &Group, size: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut chosen = Vec::with_capacity(size);
    let mut in_set = vec![false; group.n];
    while chosen.len() < size {
        let element = rng.random_range(0..group.n);
        if element == group.identity || in_set[element] {
            continue;
        }
        in_set[element] = true;
        chosen.push(element);
    }
    chosen
}

// ---------------------------------------------------------------------------
// Inverse-closed variant: regular partial difference sets / strongly regular
// Cayley graphs (SRGs). An SRG is exactly the DSRG special case t = k, but the
// connection set must be inverse-closed (S = S^-1) so its Cayley graph is
// undirected. We enforce that by moving over inverse-pairs of elements rather
// than single elements, and we search all groups (abelian included), matching
// Brouwer's SRG tables (https://aeb.win.tue.nl/graphs/srg/srgtab.html).
// ---------------------------------------------------------------------------

/// Parameters of a strongly regular graph: an inverse-closed connection set of
/// size `k` in a group of order `v` whose (undirected) Cayley graph satisfies
/// `A^2 = k*I + lambda*A + mu*(J - I - A)`. That is the DSRG equation with
/// `t = k`, so the objective and incremental scorer are shared unchanged.
pub struct SRGParameterSet {
    pub v: usize,
    pub k: usize,
    pub lambda: usize,
    pub mu: usize,
}

impl SRGParameterSet {
    /// The equivalent DSRG parameter set (`t = k`) used to drive the objective.
    fn as_dsrg(&self) -> DSRGParameterSet {
        DSRGParameterSet {
            n: self.v,
            k: self.k,
            t: self.k,
            lambda: self.lambda,
            mu: self.mu,
        }
    }
}

/// The inverse-pairs of a group's non-identity elements -- the atoms the SRG
/// search moves over. Each atom is either a proper inverse-pair `{g, g^-1}`
/// (weight 2) or a lone involution `{g = g^-1}` (weight 1). Selecting whole
/// atoms is what keeps every candidate connection set inverse-closed.
struct PairAtoms {
    /// Atom index -> its 1 or 2 element indices.
    elems: Vec<Vec<usize>>,
    /// Atom index -> weight (1 for an involution, 2 for a proper pair).
    weight: Vec<usize>,
    /// Indices (into `elems`) of the weight-1 atoms (involutions).
    involutions: Vec<usize>,
    /// Indices (into `elems`) of the weight-2 atoms (proper pairs).
    pairs: Vec<usize>,
}

impl PairAtoms {
    fn new(group: &Group) -> Self {
        let mut elems = Vec::new();
        let mut weight = Vec::new();
        let mut involutions = Vec::new();
        let mut pairs = Vec::new();
        for g in 0..group.n {
            if g == group.identity {
                continue;
            }
            let gi = group.inverse(g);
            if gi == g {
                involutions.push(elems.len());
                elems.push(vec![g]);
                weight.push(1);
            } else if g < gi {
                // Take each proper pair once, at its smaller representative.
                pairs.push(elems.len());
                elems.push(vec![g, gi]);
                weight.push(2);
            }
        }
        PairAtoms {
            elems,
            weight,
            involutions,
            pairs,
        }
    }
}

/// The involution counts `q` for which an inverse-closed set of size `k` exists:
/// `q` involutions plus `p` proper pairs give `q + 2p = k`, so `q` must share
/// `k`'s parity and leave a nonnegative, buildable number of pairs. Empty when
/// no inverse-closed connection set of size `k` fits in the group at all.
fn feasible_involution_counts(atoms: &PairAtoms, k: usize) -> Vec<usize> {
    let num_inv = atoms.involutions.len();
    let num_pairs = atoms.pairs.len();
    let mut qs = Vec::new();
    let mut q = k % 2; // smallest nonnegative q with q ≡ k (mod 2)
    while q <= k && q <= num_inv {
        let p = (k - q) / 2;
        if p <= num_pairs {
            qs.push(q);
        }
        q += 2;
    }
    qs
}

/// Draw `count` distinct atoms from `pool` uniformly without replacement via a
/// partial Fisher–Yates shuffle. `count <= pool.len()` is guaranteed by the
/// feasibility check.
fn sample_atoms(pool: &[usize], count: usize, rng: &mut StdRng) -> Vec<usize> {
    let mut buf = pool.to_vec();
    for i in 0..count {
        let j = rng.random_range(i..buf.len());
        buf.swap(i, j);
    }
    buf.truncate(count);
    buf
}

/// A random inverse-closed connection set of size exactly `k`: pick a feasible
/// involution count `q`, then `q` random involutions and `(k - q)/2` random
/// proper pairs. `feasible` must be nonempty.
fn random_pairset(
    atoms: &PairAtoms,
    feasible: &[usize],
    k: usize,
    rng: &mut StdRng,
) -> Vec<usize> {
    let q = feasible[rng.random_range(0..feasible.len())];
    let p = (k - q) / 2;
    let mut set = Vec::with_capacity(k);
    for a in sample_atoms(&atoms.involutions, q, rng) {
        set.extend_from_slice(&atoms.elems[a]);
    }
    for a in sample_atoms(&atoms.pairs, p, rng) {
        set.extend_from_slice(&atoms.elems[a]);
    }
    set
}

/// Best-improvement hill climbing over the inverse-pair neighborhood: each move
/// removes one selected atom and adds one unselected atom of the *same weight*,
/// keeping the connection set inverse-closed and of size `k`. Stops at a local
/// optimum or early on reaching error 0. `dsrg` is `params.as_dsrg()` (t = k).
fn hill_climb_srg(
    group: &Group,
    dsrg: &DSRGParameterSet,
    atoms: &PairAtoms,
    start: Vec<usize>,
) -> ScoredSet {
    let mut obj = Objective::new(group, dsrg, &start);
    if obj.error == 0 {
        return ScoredSet {
            connection_set: obj.members,
            error: 0,
        };
    }

    // An atom is selected iff its first element is in the set (atoms partition
    // the non-identity elements, so the first element decides the whole atom).
    let mut selected = vec![false; atoms.elems.len()];
    let mut selected_atoms = Vec::new();
    for (i, e) in atoms.elems.iter().enumerate() {
        if obj.in_set[e[0]] {
            selected[i] = true;
            selected_atoms.push(i);
        }
    }

    loop {
        let mut best_error = obj.error;
        let mut best_move: Option<(usize, usize)> = None; // (out_atom, in_atom)

        for si in 0..selected_atoms.len() {
            let out = selected_atoms[si];
            let w = atoms.weight[out];
            for inn in 0..atoms.elems.len() {
                if selected[inn] || atoms.weight[inn] != w {
                    continue;
                }
                let error = obj.swap_error_delta(&atoms.elems[out], &atoms.elems[inn]);
                if error == 0 {
                    obj.commit_delta(&atoms.elems[out], &atoms.elems[inn]);
                    return ScoredSet {
                        connection_set: obj.members,
                        error: 0,
                    };
                }
                if error < best_error {
                    best_error = error;
                    best_move = Some((out, inn));
                }
            }
        }

        match best_move {
            Some((out, inn)) => {
                obj.commit_delta(&atoms.elems[out], &atoms.elems[inn]);
                selected[out] = false;
                selected[inn] = true;
                let slot = selected_atoms.iter().position(|&x| x == out).unwrap();
                selected_atoms[slot] = inn;
            }
            None => {
                return ScoredSet {
                    connection_set: obj.members,
                    error: obj.error,
                };
            }
        }
    }
}

/// Random-restart hill climbing for a strongly regular Cayley graph (inverse-
/// closed / partial difference set search). Mirrors [`random_restart_hill_climb`]
/// -- same parallelism, determinism, short-circuiting, and progress accounting --
/// but every candidate is inverse-closed by construction. Returns a `ScoredSet`
/// with `error == i64::MAX` if the group admits no inverse-closed set of size `k`.
pub fn random_restart_hill_climb_srg(
    group: &Group,
    params: &SRGParameterSet,
    max_restarts: usize,
    base_seed: u64,
    progress: &AtomicUsize,
) -> ScoredSet {
    let dsrg = params.as_dsrg();
    let atoms = PairAtoms::new(group);
    let feasible = feasible_involution_counts(&atoms, params.k);
    if feasible.is_empty() {
        // No inverse-closed connection set of size k fits in this group; count
        // the whole restart budget so shared progress accounting stays honest.
        progress.fetch_add(max_restarts, Ordering::Relaxed);
        return ScoredSet {
            connection_set: Vec::new(),
            error: i64::MAX,
        };
    }

    let min_solved = AtomicUsize::new(usize::MAX);
    (0..max_restarts)
        .into_par_iter()
        .filter_map(|restart| {
            progress.fetch_add(1, Ordering::Relaxed);
            if restart >= min_solved.load(Ordering::Relaxed) {
                return None; // a lower/equal index already solves; this can't win
            }
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(restart as u64));
            let start = random_pairset(&atoms, &feasible, params.k, &mut rng);
            let result = hill_climb_srg(group, &dsrg, &atoms, start);
            if result.error == 0 {
                min_solved.fetch_min(restart, Ordering::Relaxed);
            }
            Some((restart, result))
        })
        .reduce_with(|a, b| if (b.1.error, b.0) < (a.1.error, a.0) { b } else { a })
        .map(|(_, best)| best)
        .expect("max_restarts must be at least 1")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_params() -> DSRGParameterSet {
        DSRGParameterSet {
            n: 8,
            k: 3,
            t: 2,
            lambda: 1,
            mu: 1,
        }
    }

    #[test]
    fn objective_matches_gap_reference_value() {
        // Cross-checked against GAP's e2(Diff(...)) on SmallGroup(8,1) with the
        // set {1,2,4} (0-based) and (t,lambda,mu)=(2,1,1): GAP reports 12.
        let group = Group::from_gap(8, 1).expect("gap should load SmallGroup(8,1)");
        assert_eq!(objective_error(&group, &small_params(), &[1, 2, 4]), 12);
    }

    #[test]
    fn incremental_objective_matches_full_recompute() {
        // The incremental Objective must agree with the O(k^2) reference for the
        // initial error, for every swap's error (swap_error), and after commit.
        let group = Group::from_gap(16, 3).expect("gap should load SmallGroup(16,3)");
        let params = DSRGParameterSet {
            n: 16,
            k: 6,
            t: 6,
            lambda: 2,
            mu: 2,
        };
        let mut rng = StdRng::seed_from_u64(2024);

        for _ in 0..20 {
            let start = random_subset(&group, params.k, &mut rng);
            let mut obj = Objective::new(&group, &params, &start);
            assert_eq!(obj.error, objective_error(&group, &params, &start));

            // Every neighbor's swap_error matches a fresh full recompute.
            for slot in 0..obj.members.len() {
                let g = obj.members[slot];
                for h in 0..group.n {
                    if h == group.identity || obj.in_set[h] {
                        continue;
                    }
                    let mut neighbor = obj.members.clone();
                    neighbor[slot] = h;
                    assert_eq!(
                        obj.swap_error(g, h),
                        objective_error(&group, &params, &neighbor),
                        "swap remove {g} add {h}"
                    );
                }
            }

            // Commit a few swaps and check the running error stays exact.
            for _ in 0..5 {
                let slot = rng.random_range(0..obj.members.len());
                let g = obj.members[slot];
                let mut h = rng.random_range(0..group.n);
                while h == group.identity || obj.in_set[h] {
                    h = rng.random_range(0..group.n);
                }
                obj.commit(g, h);
                assert_eq!(obj.error, objective_error(&group, &params, &obj.members));
            }
        }
    }

    #[test]
    fn random_subset_is_distinct_and_excludes_identity() {
        let group = Group::from_gap(8, 1).expect("gap should load SmallGroup(8,1)");
        let mut rng = StdRng::seed_from_u64(42);
        let subset = random_subset(&group, 5, &mut rng);
        assert_eq!(subset.len(), 5);
        assert!(!subset.contains(&group.identity));
        let mut sorted = subset.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 5, "elements must be distinct");
    }

    #[test]
    fn hill_climb_never_worsens_the_start() {
        let group = Group::from_gap(8, 1).expect("gap should load SmallGroup(8,1)");
        let params = small_params();
        let mut rng = StdRng::seed_from_u64(7);
        let start = random_subset(&group, params.k, &mut rng);
        let start_error = objective_error(&group, &params, &start);
        let result = hill_climb(&group, &params, start);
        assert!(result.error <= start_error);
        assert!(result.error >= 0);
    }

    #[test]
    fn objective_is_zero_only_for_a_genuine_solution() {
        // A^2 must reproduce t*I + lambda*A + mu*(J - I - A) exactly; a set that
        // does gets error 0, and perturbing it raises the error above 0.
        let group = Group::from_gap(8, 1).expect("gap should load SmallGroup(8,1)");
        let params = small_params();
        let progress = AtomicUsize::new(0);
        let solution = random_restart_hill_climb(&group, &params, 5000, 123, &progress);
        if solution.error == 0 {
            // Replacing one member with a different non-identity element breaks it.
            let mut broken = solution.connection_set.clone();
            let replacement = (0..group.n)
                .find(|e| *e != group.identity && !broken.contains(e))
                .unwrap();
            broken[0] = replacement;
            assert!(objective_error(&group, &params, &broken) > 0);
        }
    }

    // --- inverse-closed (SRG / partial difference set) search ---------------

    /// True iff `set` is closed under inverses (its Cayley graph is undirected).
    fn is_inverse_closed(group: &Group, set: &[usize]) -> bool {
        let members: std::collections::HashSet<usize> = set.iter().copied().collect();
        members.iter().all(|&g| members.contains(&group.inverse(g)))
    }

    #[test]
    fn pair_atoms_partition_the_nonidentity_elements() {
        // The atoms must cover every non-identity element exactly once, with
        // involutions weight 1 and proper pairs weight 2.
        let group = Group::from_gap(8, 1).expect("gap should load SmallGroup(8,1)");
        let atoms = PairAtoms::new(&group);
        let mut seen = vec![false; group.n];
        let mut total = 0;
        for (elems, &w) in atoms.elems.iter().zip(&atoms.weight) {
            assert_eq!(elems.len(), w);
            for &e in elems {
                assert_ne!(e, group.identity);
                assert!(!seen[e], "element {e} covered twice");
                seen[e] = true;
                total += 1;
            }
        }
        assert_eq!(total, group.n - 1, "atoms must cover every non-identity element");
        // Each proper pair is a genuine inverse-pair; each involution is self-inverse.
        for &p in &atoms.pairs {
            assert_eq!(group.inverse(atoms.elems[p][0]), atoms.elems[p][1]);
        }
        for &i in &atoms.involutions {
            let g = atoms.elems[i][0];
            assert_eq!(group.inverse(g), g);
        }
    }

    #[test]
    fn random_pairset_is_inverse_closed_and_sized() {
        // C13 has no involutions, so every size must be even; k=6 uses 3 pairs.
        let group = Group::from_gap(13, 1).expect("gap should load SmallGroup(13,1)");
        let atoms = PairAtoms::new(&group);
        let feasible = feasible_involution_counts(&atoms, 6);
        assert_eq!(feasible, vec![0]); // no involutions -> q must be 0
        let mut rng = StdRng::seed_from_u64(9);
        for _ in 0..20 {
            let set = random_pairset(&atoms, &feasible, 6, &mut rng);
            assert_eq!(set.len(), 6);
            assert!(is_inverse_closed(&group, &set));
            assert!(!set.contains(&group.identity));
        }
    }

    #[test]
    fn feasibility_respects_involutions_and_parity() {
        // C5 (order 5, odd) has no involutions: only even k are feasible.
        let group = Group::from_gap(5, 1).expect("gap should load SmallGroup(5,1)");
        let atoms = PairAtoms::new(&group);
        assert!(feasible_involution_counts(&atoms, 1).is_empty()); // odd k, no involutions
        assert_eq!(feasible_involution_counts(&atoms, 2), vec![0]);
        assert_eq!(feasible_involution_counts(&atoms, 4), vec![0]);
    }

    #[test]
    fn srg_search_finds_paley_pentagon_on_cyclic_group() {
        // The 5-cycle is the Paley graph SRG(5,2,0,1) as a Cayley graph on C5.
        // The search must find it, and the connection set must be inverse-closed
        // and solve the equation exactly (verified via the t=k reference).
        let group = Group::from_gap(5, 1).expect("gap should load SmallGroup(5,1)");
        let params = SRGParameterSet {
            v: 5,
            k: 2,
            lambda: 0,
            mu: 1,
        };
        let progress = AtomicUsize::new(0);
        let best = random_restart_hill_climb_srg(&group, &params, 500, 1, &progress);
        assert_eq!(best.error, 0, "should find the pentagon");
        assert!(is_inverse_closed(&group, &best.connection_set));
        assert_eq!(
            objective_error(&group, &params.as_dsrg(), &best.connection_set),
            0,
            "found set must solve A^2 = k*I + lambda*A + mu*(J-I-A)"
        );
    }

    #[test]
    fn srg_search_reports_infeasible_when_no_such_set_exists() {
        // No inverse-closed set of odd size 1 exists in involution-free C5.
        let group = Group::from_gap(5, 1).expect("gap should load SmallGroup(5,1)");
        let params = SRGParameterSet {
            v: 5,
            k: 1,
            lambda: 0,
            mu: 0,
        };
        let progress = AtomicUsize::new(0);
        let best = random_restart_hill_climb_srg(&group, &params, 10, 1, &progress);
        assert_eq!(best.error, i64::MAX);
        assert!(best.connection_set.is_empty());
        // The whole restart budget is still accounted for in progress.
        assert_eq!(progress.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn srg_hill_climb_preserves_inverse_closure() {
        // Whatever hill climbing returns (solution or local optimum) must remain
        // inverse-closed: every move swaps a whole inverse-pair for another.
        let group = Group::from_gap(13, 1).expect("gap should load SmallGroup(13,1)");
        let params = SRGParameterSet {
            v: 13,
            k: 6,
            lambda: 2,
            mu: 3,
        };
        let dsrg = params.as_dsrg();
        let atoms = PairAtoms::new(&group);
        let feasible = feasible_involution_counts(&atoms, params.k);
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..10 {
            let start = random_pairset(&atoms, &feasible, params.k, &mut rng);
            let result = hill_climb_srg(&group, &dsrg, &atoms, start);
            assert_eq!(result.connection_set.len(), params.k);
            assert!(is_inverse_closed(&group, &result.connection_set));
        }
    }
}
