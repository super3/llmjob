// refverify: the spec's own share check (zk-pow's verify_plain_proof, salted seeds, as the pools run it) over
// shares our miner made. Every other check of a share before a pool sees it is our own JS (verify-hits.js,
// verify-pool.js), so a mistake the device and the JS reference share would pass them. It can't pass this.
//
// It is not built with the app. earn/tuning/salad/amdtest.sh's refverify task fetches the spec
// (github.com/pearl-research-labs/pearl) at a pinned commit, copies this file to zk-pow/src/bin/refverify.rs
// and builds it with cargo, then runs it on the shares earlier tasks wrote.
//
// stdin: one JSON object a line, { "header": hex (76 bytes), "target": hex (32 bytes, big-endian),
// "plain_proof": base64 of the bincode (not gzipped), ... }, as verify-hits.js's and
// verify-hits-profile.js's PEARL_SHARE_DUMP and sharetap.py's `shares` write them. A line with no plain_proof
// (sharetap's summary) is skipped. The target becomes the compact nbits a pool's share check passes as
// nbits_override, rounded down as Bitcoin's GetCompact does, which can only make the check stricter.
// stdout: the first few shares that pass and the first ten that fail, a line each, then
// "refverify shares=N ok=N fail=N by_k=4096:Nok/Nfail,...". Exits 1 if any fails or none was read.
use std::collections::BTreeMap;
use std::io::{self, BufRead};

use base64::Engine;
use primitive_types::U256;
use zk_pow::api::proof::{IncompleteBlockHeader, SeedDerivation};
use zk_pow::api::verify::verify_plain_proof;
use zk_pow::ffi::plain_proof::PlainProof;

fn unhex(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 || !s.is_ascii() {
        return Err(format!("not hex: {}", s.chars().take(20).collect::<String>()));
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string())).collect()
}

// Bitcoin's GetCompact: the target's size in bytes and its top three bytes, rounded down.
fn compact(t: U256) -> u32 {
    let mut size = (t.bits() + 7) / 8;
    let mut m: u64 = if size <= 3 { t.low_u64() << (8 * (3 - size)) } else { (t >> (8 * (size - 3))).low_u64() };
    if m & 0x0080_0000 != 0 {
        m >>= 8;
        size += 1;
    }
    ((size as u32) << 24) | (m as u32 & 0x007f_ffff)
}

// One line: None to skip it, or the share's k (when it parsed) and what the verifier said.
fn check(line: &str) -> Option<(Option<usize>, Result<(), String>)> {
    let v: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return Some((None, Err(format!("not JSON: {e}")))),
    };
    let pp = v.get("plain_proof")?.as_str()?;
    let parsed = (|| -> Result<(PlainProof, IncompleteBlockHeader, u32), String> {
        let header = unhex(v.get("header").and_then(|x| x.as_str()).ok_or("no header")?)?;
        let target = unhex(v.get("target").and_then(|x| x.as_str()).ok_or("no target")?)?;
        if target.len() != 32 {
            return Err(format!("the target is {} bytes, not 32", target.len()));
        }
        let bytes = base64::engine::general_purpose::STANDARD.decode(pp).map_err(|e| format!("not base64: {e}"))?;
        let header = IncompleteBlockHeader::from_bytes(&header).map_err(|e| format!("{e:#}"))?;
        let proof = PlainProof::deserialize_compat(&bytes).map_err(|e| format!("{e:#}"))?;
        Ok((proof, header, compact(U256::from_big_endian(&target))))
    })();
    match parsed {
        Err(e) => Some((None, Err(e))),
        Ok((proof, header, nbits)) => {
            let r = verify_plain_proof(&header, &proof, Some(nbits), SeedDerivation::Salted).map_err(|e| format!("{e:#}"));
            Some((Some(proof.k), r))
        }
    }
}

fn main() {
    let mut by_k: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let (mut n, mut ok, mut bad) = (0u64, 0u64, 0u64);
    for (i, line) in io::stdin().lock().lines().enumerate() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("refverify: reading stdin: {e}");
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let Some((k, r)) = check(&line) else { continue };
        n += 1;
        let key = k.map_or_else(|| "?".to_string(), |k| k.to_string());
        let e = by_k.entry(key.clone()).or_insert((0, 0));
        match r {
            Ok(()) => {
                ok += 1;
                e.0 += 1;
                if ok <= 3 {
                    println!("ok line={} k={key}", i + 1);
                }
            }
            Err(why) => {
                bad += 1;
                e.1 += 1;
                if bad <= 10 {
                    println!("FAIL line={} k={key}: {}", i + 1, why.chars().take(300).collect::<String>());
                }
            }
        }
    }
    let ks: Vec<String> = by_k.iter().map(|(k, (o, f))| format!("{k}:{o}ok/{f}fail")).collect();
    println!("refverify shares={n} ok={ok} fail={bad} by_k={}", ks.join(","));
    std::process::exit(if bad == 0 && n > 0 { 0 } else { 1 });
}
