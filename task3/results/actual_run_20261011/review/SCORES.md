# Approved qualitative scores

Five selected answer pairs were assessed on three criteria, each scored 0–2: answering the question, support in the supplied text, and clarity/focus. For the C question, technical correctness replaced support in a supplied passage. The score table was approved on 11 October 2026.

Answer A is canonical GRPO; Answer B is the constant-denominator condition.

| Example | A /6 | B /6 |
|---|---:|---:|
| Poem/video | 2 | 1 |
| C static keyword | 4 | 4 |
| Polish sentiment | 1 | 3 |
| MidJourney instructions | 2 | 2 |
| SafeSpace passage | 2 | 2 |

Five unique prompts selected by frozen extreme reward/length differences; illustrative qualitative evidence, not representative aggregate quality.
Input truncation affects four of the five pairs; evaluate against the full original requests while preserving this limitation.

## 1. Poem/video

Prompt ID: `4e4da235589d4153d6b9614f7d069f9b986257fcb10be059f1d4366a218ea183`.

**Answer A: 2/6**

- Answers the question: 1/2. Partly discusses emotional delivery but misses historical footage.
- Supported by supplied text: 0/2. Adds unsupported title, performance details and personal commentary.
- Clear and focused: 1/2. Readable sentences with repetition, shifts of voice and an unfinished ending.

**Answer B: 1/6**

- Answers the question: 0/2. Mostly biography/publication history rather than the video question.
- Supported by supplied text: 0/2. Biographical events and publication details are unsupported by the supplied passage.
- Clear and focused: 1/2. Readable sentences, but distracting and unfinished.

## 2. C static keyword

Prompt ID: `51223958b651c1f84f3241122f66cdda11207f754b2f7feacc6fec85a3463423`.

**Answer A: 4/6**

- Answers the question: 2/2. Directly addresses static, compilation and linking.
- Technical correctness: 1/2. Correct basic storage/linkage ideas mixed with significant scope and access errors.
- Clear and focused: 1/2. Structured but repetitive and unfinished.

**Answer B: 4/6**

- Answers the question: 2/2. Directly addresses static, compilation and linking.
- Technical correctness: 1/2. Mixes correct basic ideas with scope/linkage errors, incorrect __func__/__LINE__ claims and a nonstandard nested-function example.
- Clear and focused: 1/2. Readable structure, but an unfinished and unusable code example.

## 3. Polish sentiment

Prompt ID: `c4053dc7fb2ba3bfe219c5302d384c6b714278b856b16dbe7868741ff25df843`.

**Answer A: 1/6**

- Answers the question: 1/2. Attempts classification but repeats/changes the demonstration instead of returning one word.
- Supported by supplied text: 0/2. Final Yes conflicts with the negative target review.
- Clear and focused: 0/2. Conflicting No/Yes labels and substantial extraneous text.

**Answer B: 3/6**

- Answers the question: 1/2. Provides a label but violates the required one-word output format.
- Supported by supplied text: 1/2. No fits the negative target review, with unsupported added complaints.
- Clear and focused: 1/2. Consistent No labels but unnecessary surrounding text.

## 4. MidJourney instructions

Prompt ID: `03d68fc53b5c3ee9cd4d9437e868107d87a5e4e66cfe9234cf249c4c1347dffd`.

**Answer A: 2/6**

- Answers the question: 1/2. Related image-prompt material but generates before a concept is supplied.
- Supported by supplied text: 0/2. Invents a concept and changes field meanings.
- Clear and focused: 1/2. Understandable but mixes instructions and an unsolicited example with incorrect formatting.

**Answer B: 2/6**

- Answers the question: 1/2. Related image-prompt material but rewrites instructions instead of requesting the concept.
- Supported by supplied text: 0/2. Changes atmosphere/lighting fields to actions/objects.
- Clear and focused: 1/2. Readable but begins mid-sentence and supplies unnecessary instructions.

## 5. SafeSpace passage

Prompt ID: `23bd8f99e8359f0052e16c65689eae80cb20bf937ea6ca5794d36d29734a560b`.

**Answer A: 2/6**

- Answers the question: 1/2. Discusses relaxation but omits the stated compassionate-mind mechanism.
- Supported by supplied text: 0/2. Adds unsupported institutional attribution, personalization, simulation and clinical-result claims.
- Clear and focused: 1/2. Readable but repetitive, promotional and unfinished.

**Answer B: 2/6**

- Answers the question: 1/2. Discusses relaxation but omits the stated compassionate-mind mechanism and collaborative design.
- Supported by supplied text: 0/2. Adds unsupported personalization and treatment-simulation claims.
- Clear and focused: 1/2. Readable but begins mid-sentence, drifts into event promotion and ends with an unrelated device instruction.

C technical checks use [WG14 draft N1570](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf), especially sections 6.2.1–6.2.4, 6.4.2.2, 6.8.2 and 6.10.8.1.

The received export and earlier pending-review files are retained as historical records. This approved score table is the current review outcome.
