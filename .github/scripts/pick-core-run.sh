#!/usr/bin/env bash
# Picks the native-core run whose Pearl cores this installer ships, and writes
# its id to $GITHUB_OUTPUT as run_id. Run by miner-build.yml.
#
# The cores must be built from the same native source as the commit this job
# builds. "The newest successful native-core run" did not do that: both
# workflows start on the same push, this job stages its core about a minute
# in, and native-core takes about four. So the PR build of 968115f shipped the
# core of the commit before it.
#
# 1. Use a native-core run for this commit. On a pull request, that is the
#    pull_request run for the PR head. Otherwise it is any push or
#    workflow_dispatch run for github.sha. A release tag always has one, since
#    path filters do not apply to tags, and often a second from the push to
#    main. Take the newest that succeeded. If none has yet but one is still
#    running, wait. If all of them failed or were cancelled, fail.
# 2. If there is no such run 90 s after this workflow run was created, use a
#    run built from the same native source. The candidates are the push and
#    workflow_dispatch runs on main and on this build's branch (on a pull
#    request, the branch it merges into), newest first. One counts only if
#    GitHub's compare API says its commit is in this commit's history and no
#    file changed between the two is under earn/native/ or is native-core.yml,
#    the paths that start native-core. The API lists at most 300 changed
#    files, so a list of 300 may be cut short and does not count. Take the
#    first that counts. Wait for it if it is running; fail if it failed.
# 3. If none counts but one is in this commit's history, then most likely
#    this commit changed the native source and its own run is late: wait for
#    it. If none is in this commit's history, fail.
#
# Waiting stops after 25 minutes. Needs GH_TOKEN, and PR_HEAD_SHA on
# pull_request events. Actions sets the rest.
set -euo pipefail

workflow=native-core.yml
grace=90      # seconds after this workflow run was created
timeout=1500  # 25 minutes
poll=20

api=repos/$GITHUB_REPOSITORY

now() { date +%s; }
short() { printf '%s' "${1:0:7}"; }
# To stderr, so it reaches the log from inside $(...) too. There it ends only
# the subshell; set -e then stops the script on the failed assignment.
fail() { echo "::error::$*" >&2; exit 1; }

# One retry, so a single API error does not fail a release build. Each
# attempt's output is kept apart: on an HTTP error gh prints the error body to
# stdout, and that must not end up in front of the retry's answer.
ghapi() {
  local out
  out=$(gh api "$@") || { sleep 5; out=$(gh api "$@"); } || fail "GitHub API request failed twice: $1"
  printf '%s\n' "$out"
}

deadline=$(( $(now) + timeout ))
# A re-run keeps its creation time, so the grace period is already over. That
# is fine: by then any native-core run for this commit is listed.
created=$(ghapi "$api/actions/runs/$GITHUB_RUN_ID" --jq '.created_at | fromdateiso8601')
grace_end=$(( created + grace ))

# wait_for SHA FILTER LAST: waits on the native-core runs for SHA that match
# the jq FILTER, and sets picked to the newest that succeeded. Leaves picked
# empty if there are still none at time LAST. Fails the step if they all
# failed, or on timeout. Call it as a plain command, not in an `if`: bash
# turns off set -e inside a function called from a condition, and then an API
# error would read as "no runs".
picked=
wait_for() {
  local sha=$1 filter=$2 last=$3 rows id status conclusion running failed
  while :; do
    rows=$(ghapi "$api/actions/workflows/$workflow/runs" -X GET -f head_sha="$sha" -f per_page=50 \
      --jq ".workflow_runs[] | select($filter) | [.id, .status, (.conclusion // \"none\")] | @tsv")
    running='' failed=''
    while IFS=$'\t' read -r id status conclusion; do
      [ -n "$id" ] || continue
      if [ "$status" != completed ]; then
        running=${running:-$id}
      elif [ "$conclusion" = success ]; then
        picked=$id
        return 0
      else
        failed=${failed:-"run $id ended $conclusion"}
      fi
    done <<< "$rows"

    if [ -n "$running" ]; then
      [ "$(now)" -lt "$deadline" ] \
        || fail "timed out after $(( timeout / 60 )) min waiting for native-core run $running ($(short "$sha"))"
      echo "native-core run $running for $(short "$sha") is still running; checking again in ${poll}s"
    elif [ -n "$failed" ]; then
      fail "native-core $failed for $(short "$sha"). Not taking a core from another run instead. Re-run that native-core run, then re-run this job."
    elif [ "$(now)" -ge "$last" ]; then
      return 0
    else
      echo "no native-core run for $(short "$sha") yet; checking again in ${poll}s"
    fi
    sleep "$poll"
  done
}

if [ "$GITHUB_EVENT_NAME" = pull_request ]; then
  # github.sha is the PR's merge commit. native-core's pull_request run reports
  # the PR head as its head_sha instead.
  [ -n "${PR_HEAD_SHA:-}" ] || fail "PR_HEAD_SHA must be set on pull_request events"
  sha=$PR_HEAD_SHA
  filter='.event == "pull_request"'
  why="the run for this pull request's head, $(short "$sha")"
else
  sha=$GITHUB_SHA
  filter='.event != "pull_request"'
  why="the run for $(short "$sha"), the commit this installer builds"
fi
echo "building $(short "$GITHUB_SHA") ($GITHUB_EVENT_NAME); looking for native-core runs for $(short "$sha")"
wait_for "$sha" "$filter" "$grace_end"

if [ -z "$picked" ]; then
  ref=${GITHUB_BASE_REF:-$GITHUB_REF_NAME}
  branches=main
  [ "$ref" = main ] || branches="$ref main"
  echo "no native-core run for $(short "$sha"); looking on $branches for a run built from the same native source"
  # Newest first, each commit once. pull_request runs are left out: they built
  # a merge commit, not their head_sha.
  candidates=$(for b in $branches; do
    ghapi "$api/actions/workflows/$workflow/runs" -X GET -f branch="$b" -f per_page=30 \
      --jq '.workflow_runs[] | select(.event != "pull_request") | .head_sha'
  done | awk '!seen[$0]++')

  base='' in_history=''
  for candidate in $candidates; do
    # status is "ahead" or "identical" when candidate is in this commit's
    # history. changed counts the files changed since; in_filters counts those
    # native-core's path filters cover, by new name or, for a rename, old name.
    res=$(ghapi "$api/compare/$candidate...$GITHUB_SHA?per_page=1" --jq '[.status, (.files | length),
      ([.files[] | .filename, .previous_filename | select(. != null)
        | select(startswith("earn/native/") or . == ".github/workflows/native-core.yml")] | length)] | @tsv')
    IFS=$'\t' read -r status changed in_filters <<< "$res"
    if [ "$status" != ahead ] && [ "$status" != identical ]; then
      echo "  $(short "$candidate"): not in this commit's history ($status)"
      continue
    fi
    in_history=yes
    if [ "$in_filters" = 0 ] && [ "$changed" -lt 300 ]; then
      base=$candidate
      break
    elif [ "$in_filters" != 0 ]; then
      echo "  $(short "$candidate"): in this commit's history, but $in_filters file(s) under earn/native or native-core.yml changed since"
    else
      echo "  $(short "$candidate"): in this commit's history, but $changed files changed since, too many to check"
    fi
  done

  if [ -n "$base" ]; then
    wait_for "$base" '.event != "pull_request"' 0
    [ -n "$picked" ] || fail "the native-core run for $(short "$base") is no longer listed"
    why="no native-core run for $(short "$sha"), so the run for $(short "$base"): it is in the history of $(short "$GITHUB_SHA") and has the same earn/native and native-core.yml"
  elif [ -n "$in_history" ]; then
    echo "no run in this commit's history has the same native source; waiting for a native-core run for $(short "$sha")"
    wait_for "$sha" "$filter" "$deadline"
    [ -n "$picked" ] || fail "no native-core run for $(short "$sha") after $(( timeout / 60 )) min, and none built from the same native source. Re-run this job once native-core has run for this commit."
  else
    fail "none of the last native-core runs on $branches is in the history of $(short "$GITHUB_SHA")"
  fi
fi

info=$(ghapi "$api/actions/runs/$picked" --jq '[.event, (.head_branch // "-"), .html_url] | @tsv')
IFS=$'\t' read -r event branch url <<< "$info"
echo "Pearl cores from native-core run $picked ($event on $branch): $why. $url"
echo "Pearl cores from native-core run [$picked]($url) ($event on $branch): $why." >> "$GITHUB_STEP_SUMMARY"
echo "run_id=$picked" >> "$GITHUB_OUTPUT"
