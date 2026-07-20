use std::fmt;
use std::io::Write;
use std::process::{Command, Stdio};

/// A finite group backed by its Cayley (multiplication) table.
///
/// Elements are identified with the indices `0..n`. The product `a * b` is
/// `mul(a, b)`, and `identity` is the index of the identity element. The table
/// is stored flat, row-major (`table[a * n + b]`), as `u32` (group orders here
/// are far below 2^32) so the innermost `mul` in the hot loop is a single
/// contiguous, cache-friendly load rather than a pointer chase. GAP is used only
/// to build it.
#[derive(Debug)]
pub struct Group {
    pub n: usize,
    pub identity: usize,
    table: Vec<u32>,
    /// `inv[a]` is the index of the inverse of element `a` (so
    /// `mul(a, inv[a]) == identity`). Precomputed once from the table; used by
    /// the inverse-closed (SRG / partial difference set) search to move over
    /// inverse-pairs of elements rather than single elements.
    inv: Vec<u32>,
}

#[derive(Debug)]
pub struct GapError(String);

impl fmt::Display for GapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GAP error: {}", self.0)
    }
}

impl std::error::Error for GapError {}

impl Group {
    /// Product of two elements.
    #[inline]
    pub fn mul(&self, a: usize, b: usize) -> usize {
        self.table[a * self.n + b] as usize
    }

    /// Inverse of an element: the unique `b` with `mul(a, b) == identity`.
    #[inline]
    pub fn inverse(&self, a: usize) -> usize {
        self.inv[a] as usize
    }

    /// Load a single `SmallGroup(n, i)` from GAP as a Cayley table.
    pub fn from_gap(n: u64, i: u64) -> Result<Group, GapError> {
        let text = run_gap(&dump_script(n, &format!("[{i}]")))?;
        let mut groups = parse_blocks(&text)?;
        match groups.len() {
            1 => Ok(groups.pop().unwrap().1),
            other => Err(GapError(format!("expected 1 group, GAP returned {other}"))),
        }
    }

    /// Load *every nonabelian* group of order `n` from GAP in one process, each
    /// as `(library_id, Group)`. Doing all of them in a single GAP invocation
    /// avoids paying GAP's ~1s process-startup cost once per group.
    pub fn load_nonabelian(n: u64) -> Result<Vec<(u64, Group)>, GapError> {
        let ids = format!("Filtered([1..NrSmallGroups({n})], i -> not IsAbelian(SmallGroup({n}, i)))");
        parse_blocks(&run_gap(&dump_script(n, &ids))?)
    }

    /// Load *every* group of order `n` from GAP, abelian and nonabelian alike,
    /// each as `(library_id, Group)`. Used by the inverse-closed (SRG / partial
    /// difference set) search, since strongly regular Cayley graphs arise on
    /// abelian groups too (unlike genuinely directed DSRGs).
    pub fn load_all(n: u64) -> Result<Vec<(u64, Group)>, GapError> {
        parse_blocks(&run_gap(&dump_script(n, &format!("[1..NrSmallGroups({n})]")))?)
    }
}

/// A GAP script that, for each id in the GAP list expression `ids`, prints a
/// block for `SmallGroup(n, id)`:
///
///     GROUP <id> <order> <identity_index>
///     <order rows of the Cayley table, 0-based product indices>
///
/// GAP is 1-indexed, so we subtract 1 everywhere. `Set(G)` (== `Elements(G)`,
/// a strictly sorted list) fixes the stable element ordering that all the
/// `Position` lookups agree on, and is the same enumeration the CSV schema
/// (`data/schema.md`) uses to encode `members`, so a found connection set's
/// element indices (once shifted back to 1-based) drop straight into a dpds row.
fn dump_script(n: u64, ids: &str) -> String {
    format!(
        "for i in {ids} do\n\
         \x20 G := SmallGroup({n}, i);; elts := Set(G);; nn := Size(G);;\n\
         \x20 Print(\"GROUP \", i, \" \", nn, \" \", Position(elts, Identity(G)) - 1, \"\\n\");\n\
         \x20 for a in [1..nn] do\n\
         \x20   for b in [1..nn] do\n\
         \x20     Print(Position(elts, elts[a] * elts[b]) - 1);\n\
         \x20     if b < nn then Print(\" \"); fi;\n\
         \x20   od;\n\
         \x20   Print(\"\\n\");\n\
         \x20 od;\n\
         od;\n\
         QUIT;\n"
    )
}

/// Parse a stream of `GROUP` blocks (see [`dump_script`]) into `(id, Group)`s.
fn parse_blocks(text: &str) -> Result<Vec<(u64, Group)>, GapError> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let mut groups = Vec::new();

    while let Some(header) = lines.next() {
        let mut fields = header.split_whitespace();
        if fields.next() != Some("GROUP") {
            return Err(GapError(format!("expected GROUP header, got `{header}`")));
        }
        let id: u64 = parse_field(fields.next(), "group id")?;
        let n: usize = parse_field(fields.next(), "order")?;
        let identity: usize = parse_field(fields.next(), "identity index")?;
        if identity >= n {
            return Err(GapError(format!(
                "identity index {identity} out of range for order {n} (id {id})"
            )));
        }

        let mut table = Vec::with_capacity(n * n);
        for a in 0..n {
            let row_line = lines
                .next()
                .ok_or_else(|| GapError(format!("id {id}: missing table row {a} of {n}")))?;
            let before = table.len();
            for tok in row_line.split_whitespace() {
                let entry: u32 = tok
                    .parse()
                    .map_err(|e| GapError(format!("id {id}: bad entry in row {a}: {e}")))?;
                table.push(entry);
            }
            if table.len() - before != n {
                return Err(GapError(format!(
                    "id {id}: row {a} has {} entries, expected {n}",
                    table.len() - before
                )));
            }
        }

        // Precompute inverses from the finished table: inv[a] is the b in each
        // row a whose product is the identity. Every row is a permutation, so
        // exactly one such b exists.
        let mut inv = vec![u32::MAX; n];
        for a in 0..n {
            for b in 0..n {
                if table[a * n + b] as usize == identity {
                    inv[a] = b as u32;
                    break;
                }
            }
            if inv[a] == u32::MAX {
                return Err(GapError(format!("id {id}: element {a} has no inverse")));
            }
        }

        groups.push((id, Group { n, identity, table, inv }));
    }

    Ok(groups)
}

/// Parse one whitespace-separated header field, with a descriptive error.
fn parse_field<T: std::str::FromStr>(field: Option<&str>, what: &str) -> Result<T, GapError>
where
    T::Err: std::fmt::Display,
{
    field
        .ok_or_else(|| GapError(format!("missing {what} in GROUP header")))?
        .parse()
        .map_err(|e| GapError(format!("bad {what} in GROUP header: {e}")))
}

/// Run a GAP script (via the `gap` binary) and return its stdout. The script is
/// fed on stdin and is expected to end by quitting GAP.
fn run_gap(script: &str) -> Result<String, GapError> {
    let mut child = Command::new("gap")
        .args(["-q", "-b"]) // quiet, no banner
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| GapError(format!("failed to spawn `gap`: {e}")))?;

    // GAP wraps Print output at ~80 columns by default, which splits long
    // Cayley-table rows across lines. Disabling stdout formatting keeps each
    // Print statement on one line so our line-oriented parsing holds.
    let script = format!("SetPrintFormattingStatus(\"*stdout*\", false);;\n{script}");

    child
        .stdin
        .take()
        .ok_or_else(|| GapError("could not open gap stdin".into()))?
        .write_all(script.as_bytes())
        .map_err(|e| GapError(format!("failed to write script to gap: {e}")))?;

    let output = child
        .wait_with_output()
        .map_err(|e| GapError(format!("failed to wait on gap: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(GapError(format!(
            "gap exited with {}: {}",
            output.status,
            stderr.trim()
        )));
    }

    String::from_utf8(output.stdout).map_err(|e| GapError(format!("gap output was not utf-8: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_cyclic_group_of_order_8() {
        let g = Group::from_gap(8, 1).expect("gap should load SmallGroup(8,1)");
        assert_eq!(g.n, 8);
        // Identity row/column act as identity.
        for a in 0..g.n {
            assert_eq!(g.mul(a, g.identity), a);
            assert_eq!(g.mul(g.identity, a), a);
        }
        // Table is a Latin square: each row and column is a permutation.
        for a in 0..g.n {
            let mut row: Vec<usize> = (0..g.n).map(|b| g.mul(a, b)).collect();
            row.sort_unstable();
            assert_eq!(row, (0..g.n).collect::<Vec<_>>());
        }
    }
}
