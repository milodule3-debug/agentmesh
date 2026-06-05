---
name: isus-earnings
description: >
  Calculates Dušan's ISUS teaching earnings from the Learnlight Calendar for any given month or period.
  Use this skill whenever Dušan asks about earnings, income, salary calculation, how much he made,
  payment calculation, monthly income from ISUS sessions, or anything like "calculate my earnings",
  "how much did I earn", "what are my earnings for [month]", "calculate my pay".
---

# ISUS Earnings Calculator

Calculates teaching earnings by fetching sessions from the Learnlight Calendar, classifying them by type, and applying the correct rate.

## Calendars

- **Source:** `o91ktakl1iebe8rt2e6cgcmbo3ehbsk5@import.calendar.google.com` (Learnlight Calendar)

## Rates

| Session Type | Duration | Rate |
|---|---|---|
| One-on-one | 30 min | €4.40 |
| One-on-one | 60 min | €8.80 |
| DT Groups | 45 min | €9.17 |
| Kuper | 90 min (2×45 min slots) | €21.00 per block |
| CYC VIR Teams | 90 min | €15.84 |
| Regular Groups | 60 min | €10.56 |

## Classification Rules

Apply in this order:

1. **Kuper** → name contains `kuper` → €21.00 per 90-min block. Two consecutive 45-min slots = one block (21 EUR total, not per slot).
2. **CYC VIR** → name contains `cyc vir` → €15.84 per session
3. **DT Groups** → name starts with `dt`, contains `dt `, `azubis`, or `dach` → €9.17 per session
4. **Regular Groups** → name contains any of: `bper`, `cellnex`, `pagegroup`, `3p bio`, `vir team`, `teams group`, `gruppo`, `teams en` → €10.56 per session
5. **One-on-one 30 min** → individual name, duration ≤ 35 min → €4.40
6. **One-on-one 60 min** → individual name, duration 55–65 min → €8.80

## Workflow

### Step 1 — Determine the period

If the user says "this month" or "April", use the full calendar month (1st to last day). If they say "last month", calculate accordingly. If they give a custom range, use that.

### Step 2 — Fetch all sessions

```
list_events:
  calendarId: o91ktakl1iebe8rt2e6cgcmbo3ehbsk5@import.calendar.google.com
  startTime: <first day of month>T00:00:00Z
  endTime: <last day of month>T23:59:59Z
  pageSize: 250
  orderBy: startTime
```

Page through all results if `nextPageToken` is present.

### Step 3 — Classify and calculate

For each session, apply the classification rules above, then multiply count × rate.

For **Kuper**: count the total number of 45-min Kuper slots, divide by 2 to get the number of 90-min blocks, multiply by €21.00. If there's an odd leftover slot, flag it.

### Step 4 — Report

Present a clean summary table:

| Category | Sessions | Rate | Amount |
|---|---|---|---|
| One-on-one 30 min | N | €4.40 | €X.XX |
| One-on-one 60 min | N | €8.80 | €X.XX |
| DT Groups 45 min | N | €9.17 | €X.XX |
| CYC VIR 90 min | N | €15.84 | €X.XX |
| Regular Groups 60 min | N | €10.56 | €X.XX |
| Kuper (blocks) | N | €21.00 | €X.XX |
| **TOTAL** | **N** | | **€X.XX** |

Also show total sessions and total hours worked.

## Notes

- If the user mentions a new session type or rate that doesn't fit the existing categories, ask them which rate applies and remember it for this session.
- Always double-check the Kuper block pairing logic — two 45-min Kuper slots on the same day = one block.
- CYC VIR Teams sessions are 90 min and use their own rate (€15.84), not the regular group rate.
