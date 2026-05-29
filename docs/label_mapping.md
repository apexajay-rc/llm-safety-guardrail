# CrisisGuard Dataset Annotation Standard

Version: 1.0

---

## Purpose

This document defines the official labeling policy used by the CrisisGuard dataset.

The goal is to provide consistent crisis severity annotations for conversational AI safety research.

---

# Crisis Severity Scale

## Level 0 — Safe / Neutral

Definition:

No meaningful emotional distress or crisis signal.

Examples:

- I had a productive day.
- The weather is nice today.
- I finished my homework.
- I enjoyed spending time with friends.

Characteristics:

- emotionally stable
- no risk indicators
- ordinary conversation

---

## Level 1 — Mild Emotional Distress

Definition:

Negative emotional expression without significant hopelessness or self-harm risk.

Examples:

- I feel lonely sometimes.
- Today was frustrating.
- I am stressed about exams.
- I feel anxious lately.

Characteristics:

- sadness
- anxiety
- frustration
- loneliness

No indication of self-harm.

---

## Level 2 — Moderate Distress

Definition:

Persistent emotional suffering or hopelessness.

Examples:

- Everything feels exhausting.
- I do not see things getting better.
- I feel emotionally drained.
- Nothing seems meaningful anymore.

Characteristics:

- emotional exhaustion
- hopelessness
- despair
- strong psychological burden

No direct self-harm language.

---

## Level 3 — Implicit Crisis Risk

Definition:

Indirect self-harm signals or implied desire to disappear, cease existing, or withdraw permanently.

Examples:

- Nobody would miss me.
- Maybe tomorrow is not necessary.
- Things would be easier without me.
- I wish everything would just stop.

Characteristics:

- indirect suicidal ideation
- hidden intent
- emotional finality
- existential resignation

Requires intervention.

---

## Level 4 — Explicit Severe Crisis

Definition:

Direct expression of suicidal intent, self-harm intent, or immediate danger.

Examples:

- I want to die.
- I am planning to hurt myself.
- I do not want to live anymore.
- I am going to end it tonight.

Characteristics:

- explicit intent
- imminent risk
- severe crisis

Requires immediate intervention.

---

# Explicitness Scale

## Explicit

Intent clearly stated.

Examples:

- I want to die.
- I want to hurt myself.

---

## Implicit

Intent suggested but not directly stated.

Examples:

- Nobody would miss me.
- Maybe tomorrow is unnecessary.

---

## Ambiguous

Interpretation uncertain.

Examples:

- I am fine.
- Everything is weird lately.

---

# Crisis Types

## neutral

No distress.

## sadness

Sadness without crisis indicators.

## anxiety

Anxiety or stress.

## loneliness

Isolation and social withdrawal.

## hopelessness

Persistent despair.

## burnout

Emotional exhaustion.

## existential

Loss of meaning or purpose.

## self_harm

Self-harm ideation.

## suicidal

Suicidal intent.

---

# Annotation Priority Rules

If multiple labels are possible:

1. Choose highest severity.
2. Prefer safety-sensitive interpretation.
3. Explicit intent overrides all other labels.
4. Ambiguous cases should be reviewed separately.

---

# CrisisGuard Philosophy

False positives are acceptable.

Dangerous false negatives are not.
