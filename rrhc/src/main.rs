mod group;
mod rrhc;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use clap::{ArgGroup, Args, Parser, Subcommand};
use rayon::prelude::*;

use group::Group;
use rrhc::{
    DSRGParameterSet, SRGParameterSet, ScoredSet, random_restart_hill_climb,
    random_restart_hill_climb_srg,
};

/// Search a group's Cayley table for a (directed) strongly regular Cayley graph
/// -- a connection set solving the (D)SRG equation -- by parallel random-restart
/// hill climbing. Prints `FOUND <element indices>` or `NONE` per group.
///
/// Two search modes:
///   dsrg  directed strongly regular graphs (Brouwer/Hobart), 5 parameters,
///         swept over the nonabelian groups of the order;
///   srg   strongly regular graphs (Brouwer's SRG tables) via inverse-closed
///         partial difference sets, 4 parameters, swept over *all* groups
///         (abelian included).
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Directed strongly regular graph: connection set of size k with
    /// parameters (t, lambda, mu) in a group of order n.
    Dsrg(DsrgArgs),
    /// Strongly regular graph via an inverse-closed partial difference set:
    /// connection set of size k with parameters (lambda, mu) in a group of
    /// order v. Equivalent to a DSRG with t = k, but the set is inverse-closed
    /// (its Cayley graph is undirected), searched by moving over inverse-pairs.
    Srg(SrgArgs),
}

#[derive(Args)]
struct DsrgArgs {
    /// Group order
    n: usize,
    /// Connection set size
    k: usize,
    /// DSRG parameter t
    t: usize,
    /// DSRG parameter lambda
    lambda: usize,
    /// DSRG parameter mu
    mu: usize,
    #[command(flatten)]
    common: CommonArgs,
}

#[derive(Args)]
struct SrgArgs {
    /// Group order (number of vertices v)
    v: usize,
    /// Connection set size (valency k)
    k: usize,
    /// SRG parameter lambda
    lambda: usize,
    /// SRG parameter mu
    mu: usize,
    #[command(flatten)]
    common: CommonArgs,
}

/// Options shared by both search modes: which group(s) to search plus the
/// restart budget and seed. Either give a specific `group_id`, or pass
/// `--all-groups` to sweep the whole order.
#[derive(Args)]
#[command(group(ArgGroup::new("target").required(true).args(["group_id", "all_groups"])))]
struct CommonArgs {
    /// SmallGroup library id (the i in SmallGroup(order, i))
    group_id: Option<u64>,
    /// Sweep every candidate group of the order instead of a single id
    /// (dsrg: nonabelian groups only; srg: all groups)
    #[arg(long)]
    all_groups: bool,
    /// Maximum number of hill-climbing restarts
    #[arg(short = 'r', long, default_value_t = 1000)]
    max_restarts: usize,
    /// Seed for reproducible runs; each group's RNG stream is derived from it.
    /// Omit to seed from entropy (results vary run to run).
    #[arg(long)]
    seed: Option<u64>,
}

fn main() {
    if let Err(err) = run(Cli::parse()) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

type BoxError = Box<dyn std::error::Error + Send + Sync>;

fn run(cli: Cli) -> Result<(), BoxError> {
    match cli.command {
        Command::Dsrg(args) => {
            let params = DSRGParameterSet {
                n: args.n,
                k: args.k,
                t: args.t,
                lambda: args.lambda,
                mu: args.mu,
            };
            let restarts = args.common.max_restarts;
            // DSRGs are searched on the nonabelian groups only.
            execute(args.n, &args.common, false, move |group, seed, progress| {
                random_restart_hill_climb(group, &params, restarts, seed, progress)
            })
        }
        Command::Srg(args) => {
            let params = SRGParameterSet {
                v: args.v,
                k: args.k,
                lambda: args.lambda,
                mu: args.mu,
            };
            let restarts = args.common.max_restarts;
            // SRGs (inverse-closed) arise on abelian groups too: search all.
            execute(args.v, &args.common, true, move |group, seed, progress| {
                random_restart_hill_climb_srg(group, &params, restarts, seed, progress)
            })
        }
    }
}

/// Load the target group(s), run `search` on each in parallel with live
/// progress, and print one deterministic `id=... FOUND/NONE` line per group.
///
/// `all_groups_include_abelian` selects the sweep source when `--all-groups` is
/// given: every group of the order (SRG) versus only the nonabelian ones (DSRG).
/// A specific `group_id` is always loaded directly, abelian or not.
fn execute<F>(
    order: usize,
    common: &CommonArgs,
    all_groups_include_abelian: bool,
    search: F,
) -> Result<(), BoxError>
where
    F: Fn(&Group, u64, &AtomicUsize) -> ScoredSet + Sync,
{
    if common.max_restarts == 0 {
        return Err("max_restarts must be at least 1".into());
    }

    // Load all the Cayley tables from a single GAP process up front (one GAP
    // startup, not one per group), then hand the queue to the search below.
    // Exactly one of group_id / --all-groups is set (enforced by clap).
    eprintln!("loading groups of order {order} from GAP...");
    let groups: Vec<(u64, Group)> = match common.group_id {
        Some(id) => vec![(id, Group::from_gap(order as u64, id)?)],
        None if all_groups_include_abelian => Group::load_all(order as u64)?,
        None => Group::load_nonabelian(order as u64)?,
    };
    if groups.is_empty() {
        println!("no candidate groups of order {order}");
        return Ok(());
    }

    let total = groups.len();
    let total_restarts = total * common.max_restarts;
    eprintln!(
        "loaded {total} group(s); searching (up to {} restarts each) on {} cores...",
        common.max_restarts,
        rayon::current_num_threads()
    );

    // Shared live-progress counters, printed by a monitor thread on a timer so
    // the workers never touch stderr in the hot loop.
    let restarts_done = AtomicUsize::new(0);
    let groups_done = AtomicUsize::new(0);
    let searching = AtomicBool::new(true);

    let mut results: Vec<(u64, ScoredSet)> = Vec::new();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            while searching.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(250));
                let restarts = restarts_done.load(Ordering::Relaxed);
                let pct = if total_restarts == 0 {
                    100
                } else {
                    100 * restarts / total_restarts
                };
                eprint!(
                    "\r  groups {}/{total} | restarts {restarts}/{total_restarts} ({pct}%)   ",
                    groups_done.load(Ordering::Relaxed)
                );
            }
        });

        // Workers pull groups off the queue in parallel and search each; the
        // inner restart search is parallel too (rayon shares one pool).
        results = groups
            .par_iter()
            .map(|(id, group)| {
                // Derive a well-separated per-group seed so a fixed --seed
                // reproduces regardless of core count and completion order.
                let base_seed = match common.seed {
                    Some(seed) => seed.wrapping_add(id.wrapping_mul(0x9E3779B97F4A7C15)),
                    None => rand::random::<u64>(),
                };
                let best = search(group, base_seed, &restarts_done);
                groups_done.fetch_add(1, Ordering::Relaxed);
                (*id, best)
            })
            .collect();

        searching.store(false, Ordering::Relaxed);
    });
    eprintln!(); // end the progress line

    // Final results to stdout, sorted by id: clean, pipeable, deterministic.
    results.sort_by_key(|(id, _)| *id);
    for (id, best) in &results {
        println!("id={id}: {}", summarize(best));
    }

    Ok(())
}

/// One-line summary of a result: `FOUND <indices>`, `NONE (best error N)`, or,
/// for the SRG search, `INFEASIBLE` when the group has no inverse-closed set of
/// size k at all (signalled by `error == i64::MAX`).
fn summarize(best: &ScoredSet) -> String {
    if best.error == 0 {
        let elements: Vec<String> = best.connection_set.iter().map(|e| e.to_string()).collect();
        format!("FOUND {}", elements.join(" "))
    } else if best.error == i64::MAX {
        "INFEASIBLE (no inverse-closed set of size k in this group)".to_string()
    } else {
        format!("NONE (best error {})", best.error)
    }
}
