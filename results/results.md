# Results

Train 440 requests, validation 110 (used only to set the review threshold, 0.5938), test 132 (same writing style as train), hard 55 (messy, multi-topic or very short, different wording).

## Accuracy against baselines

| Model | Test accuracy | Test macro-F1 | Hard accuracy | Hard macro-F1 |
|---|---|---|---|---|
| Keyword rules (ten-minute rules) | 56.1% | 0.549 | 56.4% | 0.554 |
| Naive Bayes (word + bigram) | 75.8% | 0.743 | 58.2% | 0.555 |
| Fine-tuned GPT-2 + LoRA | 89.4% | 0.887 | 70.9% | 0.708 |

## Review threshold: how many requests can be sorted without a human

Confidence is the probability GPT-2 gives its chosen request type (softmax over the 11 type tokens). A request below the threshold is marked NEEDS REVIEW.

| Threshold | Test: auto-sorted | Test: accuracy of those | Hard: auto-sorted | Hard: accuracy of those |
|---|---|---|---|---|
| 0.0 | 100.0% | 89.4% | 100.0% | 70.9% |
| 0.5 | 98.5% | 90.0% | 96.4% | 71.7% |
| 0.7 | 92.4% | 91.8% | 89.1% | 77.6% |
| 0.8 | 90.2% | 93.3% | 85.5% | 76.6% |
| 0.9 | 84.8% | 93.8% | 74.5% | 80.5% |
| 0.95 | 80.3% | 94.3% | 61.8% | 88.2% |
| 0.5938 (calibrated) | 96.2% | 92.1% | 94.5% | 73.1% |

## The review policy the CLI uses (confidence >= 0.9 and at least 5 words)

The adapter's calibrated threshold is 0.5938; the CLI raises it to a floor of 0.9 and always sends messages under 5 words to a human.

| Split | Calibrated threshold only: auto-sorted / accuracy | Policy: auto-sorted / accuracy | Short messages flagged |
|---|---|---|---|
| test | 96.2% / 92.1% | 84.8% / 93.8% | 0 of 132 |
| hard | 94.5% / 73.1% | 41.8% / 82.6% | 22 of 55 |

## Per-type F1 for GPT-2

| Type | Test F1 | Hard F1 |
|---|---|---|
| venue_inquiry | 0.923 | 0.571 |
| catering_request | 0.889 | 0.889 |
| decor_design | 0.88 | 0.833 |
| entertainment_booking | 0.957 | 0.571 |
| photo_video | 0.96 | 1.0 |
| guest_logistics | 0.632 | 0.571 |
| quote_pricing | 0.833 | 0.625 |
| reschedule_cancel | 1.0 | 0.8 |
| general_planning | 0.923 | 0.667 |
| complaint_feedback | 1.0 | 0.8 |
| other | 0.762 | 0.462 |

## Hard-set mistakes by GPT-2

| Request | Truth | Predicted | Confidence |
|---|---|---|---|
| hall available 14th? | venue_inquiry | other | 0.826 |
| Rooftops in Bandra? | venue_inquiry | decor_design | 0.95 |
| Is the hall big enough for cake cutting, a stage and dining all at once? How many chairs actually fit inside? | venue_inquiry | decor_design | 0.857 |
| biryani for 50? | catering_request | quote_pricing | 0.937 |
| DJ for friday? | entertainment_booking | other | 0.987 |
| Need a magician | entertainment_booking | other | 0.55 |
| Is the band available on the 22nd? The venue and food are already fixed, only the music is pending. | entertainment_booking | reschedule_cancel | 0.624 |
| Airport pickup? | guest_logistics | other | 0.948 |
| Seating for 200? | guest_logistics | quote_pricing | 0.677 |
| Cards need printing by Monday, mainly. And let me know if the printer can also do thank you notes. We'll decid | guest_logistics | quote_pricing | 0.334 |
| I paid the advance last week for decor but I've changed my mind about the theme entirely, so please drop the w | reschedule_cancel | quote_pricing | 0.669 |
| Honestly we have no idea about anything, not the venue, the food, the decor, or even the date. Someone please  | general_planning | complaint_feedback | 0.988 |
| Can I get a quick call about your services? Maybe ask about price and dates later, but first I want to underst | general_planning | other | 0.979 |
| Regarding the quote you gave, no worries, we already paid. What I want to know is why the DJ left early on Sat | complaint_feedback | quote_pricing | 0.83 |
| We are a decor supply company, not a client. Our wedding drapes and flower stands are available wholesale, ple | other | quote_pricing | 0.901 |
| Looking for a wedding gift suggestion for my cousin. Not a planner, just someone who knows something about pre | other | general_planning | 0.999 |
