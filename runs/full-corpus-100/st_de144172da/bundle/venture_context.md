# Venture context v3 — product and industry controls (unit: PITCH)

Low-inference descriptors of what is sold and to whom, coded once per full-video pass
(design: `docs/plans/design docs/2026-10-05-panel-interaction-redesign.md` §7). They are transparent
controls for the concept bottleneck models, not behavioral concepts and not an official
industry classification. Category boundaries are a pilot taxonomy to be frozen after the
development videos.

> **This Markdown file is the master copy.** The `.yaml` beside it is generated and must not be
> edited. **Provenance** is documentation only and is never sent to the model.

In these codes the residual value `NONE` means **unknown**: the footage does not
establish the category. It is kept separate from `other`.

---

# V. Venture context

## VC-INDUSTRY
- name: Primary industry
- dimension: V. Venture context
- side: none · tier: T3 · type: categorical · unit: PITCH · observability: observable
- values: food_beverage, apparel_accessories, home_household, health_personal_care, children_education, pets, sports_leisure, business_industrial_services, general_technology, other, NONE

**Definition.** The venture's primary industry, by the primary use and customer problem the
product addresses.

**Include if.** Classify from what the founders and investors say and show on screen. Software
is a product form, not an industry: a scheduling app for medical clinics is
health_personal_care or business_industrial_services by its use, not general_technology. Use
general_technology only for genuinely general-purpose technology. Additional supported
markets go in secondary industries.

**Exclude if.** Do not classify from remembered company identity, online information, investor
interest or eventual deal partners. If the primary market cannot be determined, use NONE.

**Provenance** *(never sent to the model)*. Project pilot taxonomy (design §7.1). No inherited
validity; not NAICS.

## VC-FORM
- name: Product form
- dimension: V. Venture context
- side: none · tier: T3 · type: categorical · unit: PITCH · observability: observable
- values: physical_good, software_digital, service, combined, NONE

**Definition.** What kind of thing is sold: a physical good, a software or digital product, a
service, or a combination that is sold together.

**Include if.** Classify what the customer pays for as described on screen.

**Exclude if.** An ordinary website or app that only sells a physical good does not make the
product combined. Use NONE when the footage does not establish the form.

**Provenance** *(never sent to the model)*. Project pilot taxonomy (design §7.1).

## VC-CUSTOMER
- name: Customer type
- dimension: V. Venture context
- side: none · tier: T3 · type: categorical · unit: PITCH · observability: observable
- values: consumer, business_institution, both, NONE

**Definition.** Who buys: individual consumers, businesses or institutions, or both as stated
current customers.

**Include if.** Classify from stated current sales channels and customers.

**Exclude if.** A retailer that resells to consumers does not make the customer a business;
the end buyer is the consumer. Aspirational future segments do not count. Use NONE when not
established.

**Provenance** *(never sent to the model)*. Project pilot taxonomy (design §7.1).

## VC-GENDER
- name: Presenting team gender composition
- dimension: V. Venture context
- side: founder · tier: T3 · type: categorical · unit: PITCH · observability: observable
- values: all_women, all_men, mixed, NONE

**Definition.** The gender composition of the founders who present the pitch, as they appear
and are referred to on screen. ALL_WOMEN: every presenting founder presents as a woman.
ALL_MEN: every presenting founder presents as a man. MIXED: the presenting founders include
both. NONE: it cannot be established from the footage.

**Include if.** Count only the founders who present or answer in the encounter. Use how each
person presents and how they are addressed or referred to on screen (names, pronouns, titles
used by the founders, the investors or the narration).

**Exclude if.** Do not count helpers, models, family members or employees who appear but do
not present or answer. Do not use outside knowledge of the people. This records presentation
on screen for use as a statistical control; it is not a statement of anyone's gender identity.
Use NONE when the footage leaves it unclear.

**Provenance** *(never sent to the model)*. Project control variable, added 2026-10-07.
Motivated by Kanze et al. (2018), who found investor question framing differs by entrepreneur
gender. Team-level composition is used because several pitches have more than one founder.
Perceived presentation coded from video is not self-identified gender; no inherited validity.

## VC-ASK
- name: The ask
- dimension: V. Venture context
- side: founder · tier: T3 · type: number · unit: PITCH · observability: observable

**Definition.** The investment the founders ask for in the opening and the share of the company
offered for it, as said by the founders or shown on screen: ask_amount_usd (whole US dollars)
and ask_equity_pct (per cent of equity).

**Include if.** Take the first complete statement of the ask. Give ask_equity_pct in per cent: 10
for ten per cent, never 0.10. A royalty or loan structure in the opening is recorded with the
amount as stated and equity 0, with the structure in the venture note. Record what is said, not
what is later negotiated.

**Exclude if.** Not the investors' counter-offers or the final deal. When neither spoken nor
shown, give no values with ask_missing not_observable; when the footage starts after the ask,
edited_gap.

**Provenance** *(never sent to the model)*. Project control added in v3-dev9 (2026-10-20). The
ask fixes the valuation the founders claim, the most common stated reason for a refusal; it
pairs with the deal_terms concept. Pollack, Rutherford & Nagy (2012): amount sought correlated
.82 with amount received among funded pitches.
