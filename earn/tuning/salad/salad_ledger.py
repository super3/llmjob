#!/usr/bin/env python3
# Salad's own figures for our account: the exact credit balance, every credit purchase, and what each GPU class
# and priority cost, by billing period or by day. Salad's public API has no balance or billing, so this logs in to
# the portal as the founder does ($SALAD_USER, $SALAD_PASS) and reads what the portal's billing pages read:
# the portal API for the balance, and Metronome (Salad's billing service) for purchases, invoices and daily usage.
# Read-only. Prints no password, cookie or token.
#
#   python salad_ledger.py                 balance, purchases, this period's charges by class
#   python salad_ledger.py --daily         and the cost of each UTC day
#   python salad_ledger.py --csv out.csv   a ledger: purchases and each day's charge per class, with the balance
import argparse, csv, datetime as dt, http.cookiejar, json, os, sys, urllib.error, urllib.parse, urllib.request

PORTAL = "https://portal-api.salad.com/api/portal"
GQL = "https://graphql.metronome.com/v1/graphql"
ORG = os.environ.get("SALAD_ORG", "llmjob")

# Metronome checks a query's shape strictly: a line item's fields must sit in "... on MRI_LineItem", as the
# portal's own invoice dashboard asks for them, or it answers "GraphQL Validation Failed".
INVOICES = """query RecentInvoices($limit: Int!) { Customer { id recentInvoices: mri_invoices(limit: $limit) {
  invoices { __typename id status issued_at total
    ... on ArrearsInvoice { inclusive_start_date exclusive_end_date }
    line_items { __typename ... on MRI_LineItem { display_name total }
      ... on GroupedProductChargeLineItem { sub_line_items { __typename ... on MRI_LineItem { display_name total }
        ... on ChargeLineItem { quantity unit_price } } } } } } } }"""
USAGE = """query RetrieveCustomerUsage($start_date: timestamptz!, $end_date: timestamptz!) { Customer { id
  current_usage(start_date: $start_date, end_date: $end_date) { billable_metric { name } usage { start metric_value } } } }"""


class Salad:
    def __init__(self, user, password):
        self.op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        s, j = self.call(PORTAL + "/users/login", "POST", {"email": user, "password": password})
        if s != 204:
            sys.exit(f"Salad portal login failed: HTTP {s} {str(j)[:200]}")

    def call(self, url, method="GET", body=None, headers=None):
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", "Accept": "application/json",
                                              "User-Agent": "llmjob-salad-ledger/1.0", **(headers or {})})
        try:
            with self.op.open(req, timeout=60) as r:
                t = r.read().decode()
                return r.status, json.loads(t) if t.strip() else None
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="replace")[:300]

    def balance_cents(self):
        s, j = self.call(f"{PORTAL}/organizations/{ORG}/billing-profile/credits-balance")
        if s != 200:
            sys.exit(f"Salad balance read failed: HTTP {s} {str(j)[:200]}")
        return j["amount"]

    def query(self, dashboard, q, variables):
        """A Metronome query, with the token of a fresh portal session for that billing dashboard."""
        s, j = self.call(f"{PORTAL}/organizations/{ORG}/dashboard-sessions/{dashboard}", "POST")
        if s not in (200, 201):
            sys.exit(f"Salad {dashboard} dashboard session failed: HTTP {s} {str(j)[:200]}")
        tok = urllib.parse.parse_qs(urllib.parse.urlsplit(j["embed_url"]).query)["token"][0]
        s, j = self.call(GQL, "POST", {"query": q, "variables": variables}, {"authorization": f"Bearer {tok}"})
        if s != 200 or not isinstance(j, dict) or j.get("errors"):
            sys.exit(f"Metronome {dashboard} query failed: HTTP {s} {str(j)[:300]}")
        return j["data"]["Customer"][0]


def day(s):
    return s[:10]


def usd(cents):
    return f"${cents / 100:,.2f}"


def daily_costs(charged, use):
    """Each UTC day's cost of each class: its hours that day at its invoice line's price an hour (cents). charged is
    [(period invoice, [(line name, hours, cents an hour, total)])], use Metronome's current_usage. A usage metric
    isn't always named as it's billed: on 24 Oct "RTX 5060 (16 GB) - Low" was billed as "RTX 5060 Ti (16 GB) - Low",
    and "RTX 5070 (12 GB) - Medium (Corrected)" as "RTX 5070 (12 GB) - Medium". So a metric takes the line of its
    own name, else of its name less "(corrected)", else the one line whose hours match its own over the period.
    Returns {day: {line name: (hours, cents)}}, {line name: cents an hour}, and [(metric, day, hours)] left out."""
    daily, rates, missed = {}, {}, []
    for p, lines in charged:
        hours = {}
        for m in use:
            for u in m["usage"]:
                if p["inclusive_start_date"][:10] <= day(u["start"]) < p["exclusive_end_date"][:10]:
                    hours.setdefault(m["billable_metric"]["name"].strip(), []).append((day(u["start"]), float(u["metric_value"] or 0)))
        # Metrics with no hours here take no line: "RTX 5070 (12 GB) - Low" had none, its "(corrected)" twin all.
        hours = {n: hs for n, hs in hours.items() if sum(h for _, h in hs)}
        plain = lambda n: n.lower().replace("(corrected)", "").strip()
        free, line_of = {n: (n, q, r) for n, q, r, _ in lines}, {}
        for key in (lambda n: n, plain):
            for name in hours:
                k = next((k for k in free if key(k) == key(name)), None)
                if name not in line_of and k:
                    line_of[name] = free.pop(k)
        for name, hs in hours.items():
            if name not in line_of:
                match = [k for k, (_, q, _r) in free.items() if abs(q - sum(h for _, h in hs)) < 0.02]
                if len(match) == 1:
                    line_of[name] = free.pop(match[0])
        for name, hs in hours.items():
            for d, h in hs:
                if not h:
                    continue
                if name not in line_of:
                    missed.append((name, d, h))
                    continue
                billed, _, rate = line_of[name]
                rates[billed] = rate
                was = daily.setdefault(d, {}).get(billed, (0.0, 0.0))
                daily[d][billed] = (was[0] + h, was[1] + h * rate)
    return daily, rates, missed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--daily", action="store_true", help="also print the cost of each UTC day")
    ap.add_argument("--csv", metavar="FILE", help="write a ledger: purchases and each day's charge per class")
    a = ap.parse_args()
    if not os.environ.get("SALAD_USER") or not os.environ.get("SALAD_PASS"):
        sys.exit("Set SALAD_USER and SALAD_PASS (the Salad portal login).")
    sa = Salad(os.environ["SALAD_USER"], os.environ["SALAD_PASS"])

    bal = sa.balance_cents()
    invoices = sa.query("invoice", INVOICES, {"limit": 50})["recentInvoices"]["invoices"]
    purchases = sorted((i for i in invoices if i["__typename"] == "CreditPurchaseInvoice" and i["status"] != "VOID"),
                       key=lambda i: i["issued_at"])
    periods = sorted((i for i in invoices if i["__typename"] == "ArrearsInvoice"), key=lambda i: i["inclusive_start_date"])

    # Each class and priority's hours and price an hour (cents), by billing period, from the invoices' line items.
    charged = []
    for p in periods:
        lines = []
        for li in p["line_items"]:
            for sl in li.get("sub_line_items") or []:
                if float(sl.get("total") or 0):
                    lines.append((sl["display_name"].strip(), float(sl["quantity"]), float(sl["unit_price"]), float(sl["total"])))
        charged.append((p, sorted(lines, key=lambda x: -x[3])))

    print(f"Balance: {usd(bal)}")
    print("\nCredit purchases:")
    for i in purchases:
        print(f"  {day(i['issued_at'])}  {usd(float(i['total'])):>10}")
    print(f"  {'total':10}  {usd(sum(float(i['total']) for i in purchases)):>10}")
    for p, lines in charged:
        print(f"\nCharges {day(p['inclusive_start_date'])} to {day(p['exclusive_end_date'])} ({p['status'].lower()}):")
        for name, q, rate, total in lines:
            print(f"  {name:42} {q:9.2f} h x {rate:5.1f}c  {usd(total):>9}")
        print(f"  {'total':42} {'':21} {usd(sum(x[3] for x in lines)):>9}")

    if a.daily or a.csv:
        start = min([day(i["issued_at"]) for i in purchases] + [day(p["inclusive_start_date"]) for p in periods])
        end = (dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)).isoformat()
        use = sa.query("usage", USAGE, {"start_date": start + "T00:00:00Z", "end_date": end + "T00:00:00Z"})["current_usage"]
        daily, rates, missed = daily_costs(charged, use)
        for name, d, h in missed:
            print(f"  (no invoice line for {name}: {h:.2f} h on {d} left out)")
        if a.daily:
            print("\nCost by UTC day:")
            for d in sorted(daily):
                print(f"  {d}  {usd(sum(c for _, c in daily[d].values())):>9}")
            print(f"  {'total':10}  {usd(sum(c for v in daily.values() for _, c in v.values())):>9}"
                  "  (the invoices' total above should match)")
        if a.csv:
            rows = [(day(i["issued_at"]), "purchase", "credit", "", "", float(i["total"])) for i in purchases]
            rows += [(d, "charge", n, f"{h:.4f}", f"{rates[n]:g}", -c) for d, v in daily.items() for n, (h, c) in v.items()]
            run = 0.0
            with open(a.csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["date_utc", "type", "item", "hours", "cents_per_hour", "amount_usd", "balance_usd"])
                for r in sorted(rows, key=lambda r: (r[0], r[1] != "purchase", r[2])):
                    run += r[5]
                    w.writerow([*r[:5], f"{r[5] / 100:.4f}", f"{run / 100:.4f}"])
            print(f"\nWrote {len(rows)} rows to {a.csv}; it ends at {usd(run)} against the balance of {usd(bal)}.")


if __name__ == "__main__":
    main()
