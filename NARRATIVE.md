# NARRATIVE — engineering decisions behind this project

This file records the *reasoning*, not the results. Numbers live in the code
output and in the README; the arguments that produced them live here.

Written incrementally, one section per phase.

---

## Phase 3 — What does "failure" actually mean, and when is a score honest?

### 1. The AI4I label is an instantaneous label, and real problems rarely are

In this dataset `Machine failure` marks the row on which the machine broke.
That is a perfectly usable teaching label, but it is not the label a real
predictive-maintenance system is trained on, because a system that tells you
the machine is broken *at the moment it breaks* has no value. The maintenance
crew already knows.

A real project has to choose a **prediction horizon** — how far ahead the
warning must arrive — and then rewrite the label accordingly:

```
y = 1  if a failure occurs within the next H hours, else 0
```

The horizon is not a modelling parameter. It is an operations parameter, and
it is fixed by two facts about the plant:

* **Lead time.** How long does it take to actually act? If the spare part has
  a 48-hour supplier lead time, a 6-hour horizon produces alarms nobody can
  respond to. The horizon must be at least as long as (part availability +
  crew scheduling + the machine's safe stopping procedure).
* **Signal life.** How early does the degradation become visible in the
  sensors at all? Pushing the horizon out to 30 days when bearing wear only
  becomes measurable 3 days before seizure means the first 27 days of every
  positive window are labelled "about to fail" while looking identical to
  healthy operation. That teaches the model nothing and floods it with
  contradictory examples.

The horizon therefore lands where those two windows overlap, and it is
negotiated with maintenance planning, not chosen by the data scientist.

### 2. Changing the horizon changes the problem, not just the score

Three consequences follow directly, and they are worth stating because they
are the difference between someone who has read about labelling and someone
who has done it:

1. **The positive class grows with H.** One failure event stops being one
   positive row and becomes every row inside its window. A 24-hour horizon on
   hourly data turns one event into 24 positives. Class imbalance improves —
   but the extra positives are not independent observations, they are 24 views
   of the same event. Any evaluation that treats them as independent
   overstates how much evidence there is.
2. **Precision and recall change meaning.** With an instantaneous label,
   "false positive" means the model cried wolf on a healthy row. With a
   horizon label, a warning fired 3 hours outside the window is counted as a
   false positive even though it identified a genuinely degrading machine.
   Serious systems therefore stop scoring rows and start scoring **events**:
   was each real failure preceded by at least one alarm, and how many alarm
   episodes were there in total?
3. **Splitting stops being optional.** Overlapping windows guarantee that
   neighbouring rows share information. A random split then puts hour 7 of an
   event in training and hour 8 in test, and the reported score measures
   interpolation inside a known event rather than the detection of an unknown
   one. This is the same failure Experiment B of this phase demonstrates, in a
   more severe form.

### 3. Maintenance resets the machine, and the data must say so

After a repair the machine is not the machine it was one row earlier. A
replaced tool has zero wear; a replaced bearing has zero accumulated fatigue.
Any cumulative feature is therefore only meaningful **relative to the last
intervention**, never since the beginning of the log.

This dataset shows the mechanism plainly. `Tool wear` rises on 98.8% of
consecutive rows and falls on the remaining 1.2% — those falls are the tool
changes. The correct unit of analysis is not the machine and not the row; it
is the **maintenance cycle**: the stretch of operation between two
interventions.

Three rules follow:

* **Cumulative features are counted from the last reset,** and the reset
  events have to come from the maintenance record. Where no such record
  exists, the resets must be detected (a large negative jump in a counter that
  otherwise only rises) and that detection is itself a modelling assumption
  worth documenting.
* **Splits are made at cycle boundaries,** so that a single cycle is never
  divided between training and test.
* **Preventive replacements are censored observations, not healthy ones.** A
  tool changed at 200 minutes on schedule did not survive to 250 minutes — we
  simply stopped watching. Labelling that stretch "healthy" teaches the model
  that 200 minutes of wear is safe, which is precisely the claim the data
  cannot support. This is standard survival analysis, and ignoring it biases
  the model toward optimism about exactly the machines that were nursed most
  carefully.

### 4. Why the honest number in this repo is the low one

Two experiments in `src/phase3_leakage.py` produce the same lesson from
different directions.

Giving the model the failure-mode columns (`TWF/HDF/PWF/OSF/RNF`) lifts recall
to 0.97 with perfect precision. Those columns are the maintenance technician's
diagnosis, written after the breakdown. On the shop floor they are empty while
the part is still being cut. The 0.97 is arithmetically correct and
operationally meaningless.

Splitting by row order instead of at random costs another 9 points of recall —
but that particular 9 points is not a finding, and saying so matters more than
quoting it. The ordered test set holds 39 failures, so one failure is worth
2.56 points of recall, and the Wilson 95% intervals of the two estimates
([0.199, 0.411] against [0.108, 0.355]) overlap across most of their range.
The drop is inside the noise.

What the ordered split does establish, on counts large enough to trust, is
something the random split hides entirely: the last 20% of the production run
contains 39 failures where the first 80% contains 300 — a failure rate of
1.95% against 3.75%. The process drifted. A shuffled split averages that drift
away and reports a model evaluated on a period that, in the real timeline, no
longer exists.

The distinction is the point. A difference that may be real but is
unmeasurable at this sample size gets reported as unmeasurable; a difference
counted on 339 events gets reported as fact. Experiment A needs no such
caveat — 66 caught against 20 on the identical test set is beyond any
interval's reach.

The number this project reports is therefore the pessimistic one, on purpose.
A predictive-maintenance model is bought on the promise of a specific number
of avoided stoppages. A score obtained by peeking is a promise that will be
audited by the plant manager six months later, and it will not survive.
