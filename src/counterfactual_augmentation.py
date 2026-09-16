"""
Counterfactual Data Augmentation (CDA) targeting two specific, measured
failure patterns found during live API testing (see chat), not a
wholesale retrain:

1. Insult-severity non-monotonicity: harsher personal insults sometimes
   scored LOWER on Hate Speech than milder ones (e.g. "you're an idiot"
   scored higher than "a worthless piece of garbage"). The model has
   learned specific lexical triggers rather than a coherent notion of
   severity. We add graded-severity personal-insult examples, ALL
   correctly labeled Offensive Language (not Hate Speech, since none
   target a protected group), to teach that personal insults belong in
   that class regardless of harshness.

2. Identity-term confidence instability: "gay" and "white" showed far
   larger confidence swings on neutral, benign sentences than 8 other
   identity terms tested identically (14-31 percentage points vs <3pp for
   the rest, on "I have a [term] friend and we had lunch today"-style
   sentences). This is exactly the failure mode CDA was designed for:
   augment training data with clearly benign sentences containing the
   term, so its mere presence stops correlating with offensiveness.

We ALSO add a small number of genuine group-targeted hate examples,
labeled Hate Speech -- augmenting ONLY with "insults aren't hate speech"
examples risks teaching the opposite bad lesson (that nothing sufficiently
harsh IS hate speech). Keeping both directions preserves the actual
Hate-Speech/Offensive distinction the dataset's annotation scheme intends.
Kept generic (a hypothetical "that group"/"they" framing) rather than
targeting any specific real protected group with graphic content -- the
goal is preserving the real distinction during training, not generating
maximally shocking content.
"""

# --- Graded-severity personal insults: all Offensive Language (1), NOT
# Hate Speech (0). None target a protected group -- personal insults only,
# varying only in harshness, to teach severity != category change. ---
# --- Graded-severity personal insults: all Offensive Language (1), NOT
# Hate Speech (0). None target a protected group -- personal insults only,
# varying only in harshness, to teach severity != category change.
# Expanded from the original 16 after held-out testing showed the first
# version had no measurable effect -- 16 examples out of 17,453 training
# rows (0.09%) was likely too diluted a signal. More examples here, and
# prepare_augmented_training_data.py now oversamples this specific group. ---
SEVERITY_AUGMENTATION = [
    ("you're kind of annoying sometimes", 1),
    ("you can be pretty irritating honestly", 1),
    ("you're a bit clueless sometimes", 1),
    ("you're an idiot", 1),
    ("you're such a moron", 1),
    ("you're an absolute fool", 1),
    ("shut up you absolute idiot", 1),
    ("you're a complete loser", 1),
    ("you're a worthless piece of garbage", 1),
    ("you're pathetic and everyone knows it", 1),
    ("you're utterly repulsive as a person", 1),
    ("I hate you and everything you stand for", 1),
    ("I despise every single thing about you", 1),
    ("you're the most disgusting person I've ever met", 1),
    ("you're the worst human being alive", 1),
    ("get lost, nobody wants you here", 1),
    ("you're a waste of space", 1),
    ("you're too stupid to understand anything", 1),
    ("I can't stand how dumb you are", 1),
    ("you make me sick, you total jerk", 1),
    ("everyone thinks you're an embarrassment", 1),
    ("you're a genuinely terrible person", 1),
    ("I can't believe how awful you are", 1),
    ("you make everyone around you miserable", 1),
]

# --- Genuine group-targeted hate speech: Hate Speech (0). ---
HATE_SPEECH_AUGMENTATION = [
    ("people like them don't deserve to live among us", 0),
    ("that entire group is subhuman filth that should be wiped out", 0),
    ("they should all be rounded up and gotten rid of permanently", 0),
    ("that kind of people are a disease infecting our country", 0),
    ("every single one of them deserves to suffer for existing", 0),
    ("we need to eliminate their whole community, they are vermin", 0),
]

# --- Benign identity-term sentences: Neither (2). Multiple templates so
# the model doesn't just memorize one sentence pattern. ---
IDENTITY_TERMS = [
    "gay", "white", "black", "muslim", "jewish", "christian",
    "trans", "disabled", "mexican", "asian", "latino", "queer",
]
IDENTITY_TEMPLATES = [
    "I have a {term} friend and we had lunch today",
    "my {term} coworker just got a big promotion",
    "she is {term} and works as a doctor downtown",
    "my neighbor is {term} and always waters my plants when I'm away",
    "he's {term} and just adopted the cutest puppy",
    "our {term} teacher is so kind and patient with everyone",
    "I met a {term} person at the conference who was really inspiring",
]


def build_identity_augmentation():
    rows = []
    for term in IDENTITY_TERMS:
        for template in IDENTITY_TEMPLATES:
            rows.append((template.format(term=term), 2))
    return rows


def build_full_augmentation_set():
    return list(SEVERITY_AUGMENTATION) + list(HATE_SPEECH_AUGMENTATION) + build_identity_augmentation()
