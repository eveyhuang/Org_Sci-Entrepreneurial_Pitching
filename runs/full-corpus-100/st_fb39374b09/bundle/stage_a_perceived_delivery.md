# Stage A v2 — perceived presentation delivery (provisional)

The Markdown is the master; YAML is generated. Primary outcome: pitch_positivity.
Secondary dimensions: expressed_enthusiasm and interpersonal_warmth. No composite or PCA
is prescribed. All operationalizations are provisional and require human validation.

Read [the shared human–model protocol](../prompts/stage_a_perceived_delivery_protocol.md).
The model prompt includes that protocol verbatim; it is part of the coding task.

## pitch_positivity
- dimension: Overall perceived positivity
- side: founder · tier: T2 · type: ordinal · scale: 1–5 · unit: PITCH · observability: interpretive

**Definition.** The overall positive, enthusiastic impression communicated by the presenting team across the prepared presentation. This is a direct observer judgment of displayed delivery, not an average of other ratings or an assessment of the venture.

**Scale**
1. Little or no positive expression. Delivery is largely detached, flat or negatively toned; positive engagement is scarcely communicated. This does not imply a bad venture.
2. Slight positive expression. Occasional positive engagement is apparent, but much of the presentation remains matter-of-fact or subdued.
3. Moderate positive expression. Clear positive engagement recurs alongside substantial neutral stretches.
4. Strong positive expression. Positive, enthusiastic engagement characterizes most of the presentation, with some neutral or mixed moments.
5. Very strong positive expression. The presentation communicates pronounced, sustained positive engagement, while allowing brief neutral moments.

**Include if.** Apply the shared Stage A perceived-delivery protocol: rate the presenting team's displayed performance in the supplied prepared-presentation clip using audiovisual evidence, with contributions considered over founder speaking time. Both audible founder speech and usable views must permit the intended judgment; otherwise mark uncodable.

**Exclude if.** Venture quality, optimism of business forecasts alone, attractiveness, production quality or inferred funding success. Do not derive this score by averaging enthusiasm and warmth. A calm presentation is not automatically negative.

**Provenance** *(never sent to the model)*. Project-authored translation inspired by Hu & Ma (2025), sections II.D.2–D.3: their composite Pitch Factor is compared with human judgments of overall positivity, defined for raters as enthusiasm and passion. These anchors and this team-level protocol are new and unvalidated; this is neither their exact human survey nor their ten-feature estimator. Restored 2026-10-20 as the Stage A book of release v3-dev9 (refocus plan §3.5); the ten-item book of v3-dev8 is archived in docs/codebooks/archive/stage_b_panel_v3dev8/.

## expressed_enthusiasm
- dimension: Expressed enthusiasm
- side: founder · tier: T2 · type: ordinal · scale: 1–5 · unit: PITCH · observability: interpretive

**Definition.** The strength of engaged excitement about the venture communicated through the presentation. Integrate delivery, expression, gesture and meaningful engagement with the subject; judge expressed enthusiasm, not the founder's actual feelings.

**Scale**
1. Little or no expressed excitement. The founder delivery remains detached or purely matter-of-fact.
2. Slight expressed excitement. Brief moments of interest or animation occur without a sustained sense of engaged excitement.
3. Moderate expressed excitement. Engaged animation recurs at meaningful points but alternates with subdued delivery.
4. Strong expressed excitement. Sustained engagement with the subject is evident across much of the presentation, even if expressed quietly.
5. Very strong expressed excitement. Pronounced, sustained engagement and excitement characterize the presentation across its substantive content.

**Include if.** Apply the shared Stage A perceived-delivery protocol: rate the presenting team's displayed performance in the supplied prepared-presentation clip using audiovisual evidence, with contributions considered over founder speaking time. Both audible founder speech and usable views must permit the intended judgment; otherwise mark uncodable.

**Exclude if.** Loudness, speed, smiling, positive adjectives or dramatic gestures alone. Warmth toward people without excitement about the venture. Nervous activation without engaged excitement. Do not infer genuine inner passion.

**Provenance** *(never sent to the model)*. Project-authored perceptual dimension motivated by the enthusiasm/passion interpretation of Hu & Ma (2025). It is not their vocal-arousal feature, an internal entrepreneurial-passion scale, or an inherited validated measure. Expected overlap with the primary positivity item must be evaluated, not hidden by aggregation. Restored 2026-10-20 as the Stage A book of release v3-dev9 (refocus plan §3.5); the ten-item book of v3-dev8 is archived in docs/codebooks/archive/stage_b_panel_v3dev8/.

## interpersonal_warmth
- dimension: Interpersonal warmth
- side: founder · tier: T2 · type: ordinal · scale: 1–5 · unit: PITCH · observability: interpretive

**Definition.** How friendly, approachable and positively oriented toward other people the presenting team comes across in this presentation, integrating delivery and the contextual meaning of speech. This measures perceived interpersonal expression, not trustworthiness or moral character.

**Scale**
1. Little or no communicated warmth. Presentation comes across as impersonal, distant or dismissive rather than approachable.
2. Slight communicated warmth. Brief courteous or friendly moments occur within mostly impersonal delivery.
3. Moderate communicated warmth. An approachable, friendly orientation is apparent at several points alongside impersonal stretches.
4. Strong communicated warmth. Friendly, approachable and considerate expression characterizes much of the presentation.
5. Very strong communicated warmth. A pronounced and sustained approachable, considerate orientation toward others characterizes the presentation.

**Include if.** Apply the shared Stage A perceived-delivery protocol: rate the presenting team's displayed performance in the supplied prepared-presentation clip using audiovisual evidence, with contributions considered over founder speaking time. Both audible founder speech and usable views must permit the intended judgment; otherwise mark uncodable.

**Exclude if.** Excitement or high energy alone; attractiveness; merely mentioning community, customers or a charitable mission; inferred honesty or competence. A quiet or serious presenter can convey warmth. Reciprocity is not required here: dyadic rapport belongs to Stage B.

**Provenance** *(never sent to the model)*. Project-authored holistic perceptual operationalization. Hu & Ma (2025) include dictionary-derived verbal warmth, which is a related but different measure. This code integrates audiovisual expression and context; its anchors have no inherited validation. It is distinct from Stage B reciprocal affiliation. Restored 2026-10-20 as the Stage A book of release v3-dev9 (refocus plan §3.5); the ten-item book of v3-dev8 is archived in docs/codebooks/archive/stage_b_panel_v3dev8/.

