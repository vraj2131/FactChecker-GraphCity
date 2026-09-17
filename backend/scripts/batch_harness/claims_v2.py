"""Second 200-claim corpus — a follow-up evaluation run, disjoint from
claims.py (the first 200-claim run, August 2026).

Every claim here is new: none repeats a claim from claims.py, nor from the
earlier 15-claim ad-hoc myth batch tested in the same session. A dedicated
"Common Myths & Misconceptions" domain (40 claims) was added specifically to
stress-test the quantifier-aware / numeric-hint improvements shipped after
the first run, since myths are exactly the shape those changes targeted.

Claims are interleaved round-robin across domains at execution time (see
ordered_claims()), same rationale as the first run: NewsAPI's 100/day free
cap dies early, so whichever claims run first should span every domain
rather than concentrate in one.
"""
from typing import Dict, List, Tuple

CLAIMS_BY_DOMAIN: Dict[str, List[str]] = {
    "Common Myths & Misconceptions": [
        "Toads or frogs give you warts if you touch them.",
        "Poinsettia plants are highly poisonous if a person or pet eats a leaf.",
        "Police require a 24-hour waiting period before they can investigate a missing person.",
        "Urinating on a jellyfish sting relieves the pain.",
        "The Coriolis effect determines which direction water drains in a sink or toilet.",
        "People are strictly left-brained or right-brained depending on whether they are more logical or creative.",
        "A duck's quack does not produce an echo.",
        "Camels store water in their humps.",
        "Standing near a running microwave oven exposes you to harmful radiation leakage.",
        "Fortune cookies originated in ancient China.",
        "Fish do not feel pain the way mammals and birds do.",
        "Fingernails need to \"breathe\" and are damaged by wearing nail polish for long periods.",
        "An electric fan running overnight in a sealed room can cause a person to suffocate.",
        "Planting marigolds around a garden repels all harmful insects.",
        "Poison ivy rashes can spread across the body by scratching the affected skin.",
        "Swimming immediately after eating a meal causes severe cramps and drowning.",
        "The far side of the Moon never receives any sunlight.",
        "Humans are born with an instinctive fear of snakes and spiders.",
        "A patient's body can develop antibiotic resistance during a single course of treatment.",
        "An adult's core personality traits are permanently fixed and cannot meaningfully change.",
        "Chewing celery burns more calories than the celery itself contains.",
        "The Vikings were the first Europeans to reach North America.",
        "Lemmings will follow each other off cliffs in mass suicidal migrations.",
        "Different species can never interbreed to produce fertile offspring under any circumstances.",
        "Touching a baby bird will cause its mother to abandon it because of the human scent.",
        "Static electric shocks in winter are caused by dry air alone, regardless of clothing material.",
        "The five-second rule makes food safe to eat if picked up quickly enough after being dropped.",
        "Earwigs commonly crawl into people's ears while they sleep.",
        "Humans and dinosaurs coexisted at the same time on Earth.",
        "Cold showers are more effective than vaccines at boosting the immune system.",
        "Drinking alcohol effectively warms up the body during cold weather.",
        "Ancient Egyptians used slave labor exclusively to build the pyramids.",
        "Sharks must keep swimming constantly or they will suffocate and die.",
        "A person's IQ score is fixed at birth and cannot change throughout life.",
        "Peanuts are classified botanically as nuts.",
        "The human tongue is the strongest muscle in the body relative to its size.",
        "Ostrich chicks can outrun a galloping horse from the moment they hatch.",
        "A goldfish kept in a small bowl will only grow as large as its container allows.",
        "Swallowing watermelon seeds will cause a watermelon to grow inside your stomach.",
        "The Sahara Desert was always a vast expanse of sand dunes throughout ancient history.",
    ],
    "Science — Physics & Chemistry": [
        "Diamonds are formed exclusively from compressed coal deep underground.",
        "Helium makes your voice sound higher because it is denser than air.",
        "Glass shatters more easily at extremely low temperatures than at room temperature.",
        "Mixing bleach and ammonia produces a toxic gas.",
        "The freezing point of pure water at sea level is exactly 0 degrees Celsius.",
        "Static electricity can be safely ignored around fuel pumps.",
        "Aluminum is more abundant in Earth's crust than iron.",
        "Neon signs are filled exclusively with neon gas regardless of their color.",
        "A vacuum completely blocks the transfer of heat by all methods.",
        "Table salt is a compound made of sodium and chlorine atoms.",
        "Copper is a better conductor of electricity than silver.",
    ],
    "Science — Biology": [
        "Octopuses are capable of changing the texture of their skin, not just its color.",
        "The human body contains more bacterial cells than human cells.",
        "Elephants are the only mammals that cannot jump.",
        "Koalas sleep for around 20 hours a day.",
        "Human tooth enamel is the hardest substance in the human body.",
        "All spiders produce venom that is dangerous to humans.",
        "Sea otters use rocks as tools to crack open shellfish.",
        "Human red blood cells lack a nucleus when mature.",
        "Kangaroos cannot walk backwards.",
        "The liver is the only human organ capable of regenerating lost tissue.",
        "Praying mantises always eat their mate's head during mating.",
    ],
    "Health & Medicine": [
        "Ibuprofen and acetaminophen work through the same biological mechanism.",
        "The human body's core temperature is exactly 98.6 degrees Fahrenheit for everyone.",
        "Insulin was the first hormone to be chemically synthesized in a laboratory.",
        "Smallpox is the only human disease to have been fully eradicated worldwide.",
        "Type 1 diabetes is caused primarily by diet and lifestyle choices.",
        "The BCG vaccine was developed primarily to prevent tuberculosis.",
        "Human blood types are determined by antigens present on red blood cells.",
        "Morphine is derived from the opium poppy plant.",
        "The common cold and influenza are caused by the same family of viruses.",
        "Regular exercise has been shown to reduce the risk of several types of cancer.",
        "X-rays were discovered by Wilhelm Conrad Röntgen.",
    ],
    "History": [
        "The printing press was invented by Johannes Gutenberg in the 15th century.",
        "The Roman Empire officially split into Western and Eastern halves in 285 CE under Diocletian's reforms.",
        "The Wright brothers' first powered flight took place in 1903.",
        "The Suez Canal was completed and opened in 1869.",
        "King Tutankhamun's tomb was discovered by Howard Carter in 1922.",
        "The Boston Tea Party took place in 1773.",
        "The Ottoman Empire lasted for more than 600 years.",
        "Anne Frank's diary was published after being found by her father.",
        "The first modern Olympic marathon distance was standardized at the 1908 London Games.",
        "The Black Death killed the majority of Europe's population in the 14th century.",
        "Ancient Sparta was governed by a single, all-powerful king.",
    ],
    "Politics & Government": [
        "The United States has a bicameral national legislature.",
        "Germany's head of government is called the Chancellor.",
        "The International Criminal Court has the power to prosecute any country's leader without exception.",
        "South Africa held its first fully democratic elections in 1994.",
        "The Vatican is the smallest independent state to hold a seat at the United Nations General Assembly.",
        "Japan's constitution renounces the right to maintain offensive military forces.",
        "The European Parliament is directly elected by citizens of EU member states.",
        "Every member of the United Nations Security Council holds veto power.",
        "Iceland has not maintained a standing army for decades.",
        "The president of France is elected to a five-year term.",
        "All fifty US states require a majority popular vote to award their electoral votes.",
    ],
    "Economics & Finance": [
        "The London Stock Exchange is one of the oldest stock exchanges in the world.",
        "Cryptocurrency transactions are always completely anonymous and untraceable.",
        "The gold standard was fully abandoned by the United States in 1971.",
        "Venture capital funding is a primary source of financing for most small businesses globally.",
        "The Consumer Price Index is commonly used to measure inflation.",
        "Central banks can never lose money by printing more currency.",
        "The Bretton Woods Agreement established the International Monetary Fund.",
        "Diversifying an investment portfolio eliminates all financial risk.",
        "Japan experienced a prolonged period of economic stagnation often called the \"Lost Decade.\"",
        "Credit scores are calculated identically by every credit bureau.",
        "Adam Smith is widely regarded as a founding figure of modern economics.",
    ],
    "Geography & Travel": [
        "Russia is the largest country in the world by total land area.",
        "The Amazon River carries more water than any other river on Earth.",
        "Canada has more freshwater lakes than any other country.",
        "Mount Everest's summit is the point farthest from the Earth's core.",
        "The Panama Canal connects the Atlantic and Pacific Oceans.",
        "Monaco is the second-smallest country in the world by land area.",
        "New Zealand is located directly southeast of Australia.",
        "The Amazon rainforest lies entirely within Brazil's borders.",
        "Chile is the longest country in the world from north to south.",
        "The Strait of Gibraltar separates Europe from Africa.",
        "Death Valley holds the record for the highest air temperature ever reliably recorded on Earth.",
    ],
    "Space & Astronomy": [
        "Mercury is the closest planet to the Sun.",
        "A shooting star is typically a small piece of interplanetary debris burning up in the atmosphere.",
        "The International Space Station orbits Earth roughly every 90 minutes.",
        "Uranus rotates on an axis that is tilted almost sideways relative to its orbit.",
        "Betelgeuse is expected to eventually explode as a supernova.",
        "The Hubble Space Telescope was launched in 1990.",
        "Mars has two small moons named Phobos and Deimos.",
        "Solar eclipses can only occur during a new moon phase.",
        "The asteroid belt lies between the orbits of Mars and Jupiter.",
        "A light-year is a measurement of time rather than distance.",
        "Saturn's density is low enough that it would float in water.",
    ],
    "Computer Science & Technology": [
        "TCP guarantees reliable, ordered delivery of data packets across a network.",
        "The first practical web browser was called Mosaic.",
        "Binary code represents data using only the digits 0 and 1.",
        "A compiler and an interpreter perform the exact same function in software execution.",
        "The C programming language was developed at Bell Labs.",
        "Public-key cryptography requires both parties to share the same secret key in advance.",
        "USB flash drives use magnetic storage to retain data.",
        "Assembly language provides a direct one-to-one mapping to a specific processor's machine instructions.",
        "Wikipedia is edited exclusively by a small paid staff of professional writers.",
        "Open-source software can never be sold commercially.",
        "The Turing test was proposed as a way to evaluate machine intelligence.",
    ],
    "Social Media & Online Controversy": [
        "Snapchat pioneered the disappearing-message format before other major platforms adopted it.",
        "Pinterest was originally launched as an invite-only platform.",
        "The first tweet ever posted was sent by Twitter co-founder Jack Dorsey.",
        "Discord was originally created primarily as a tool for gamers to communicate.",
        "Wikipedia's content is licensed under a copyleft license that permits reuse.",
        "Clubhouse's audio-only format required an iPhone at its initial launch.",
        "Vine was shut down entirely before TikTok's short-video format gained popularity.",
        "All content posted publicly on social media automatically enters the public domain.",
        "YouTube's algorithm relies solely on view counts to rank recommended videos.",
        "Telegram offers end-to-end encryption by default in all standard one-on-one chats.",
    ],
    "Environment & Climate": [
        "Coral bleaching is primarily caused by rising ocean water temperatures.",
        "Methane is a more potent greenhouse gas than carbon dioxide over a 20-year period.",
        "The Kyoto Protocol legally bound all United Nations member countries to emissions targets.",
        "Deforestation rates in the Amazon have remained completely constant since 1970.",
        "Renewable energy sources like solar and wind cannot be stored using any current battery technology.",
        "Permafrost thaw can release stored greenhouse gases into the atmosphere.",
        "The hole in the ozone layer was primarily caused by chlorofluorocarbon emissions.",
        "Composting food waste produces zero methane emissions.",
        "Global plastic production has increased significantly since the mid-20th century.",
        "Fracking has no documented connection to increased seismic activity in any region.",
    ],
    "Sports": [
        "The Iditarod is a long-distance sled dog race held in Alaska.",
        "A regulation FIFA soccer match lasts 90 minutes of play before stoppage time.",
        "Table tennis originated in England as an indoor version of lawn tennis.",
        "The Ryder Cup is a golf competition between teams from the United States and Europe.",
        "Wimbledon is the oldest tennis tournament in the world.",
        "Ice hockey periods in the NHL are each 20 minutes long.",
        "The Boston Marathon's course is certified downhill, making all its times ineligible for world records.",
        "Muhammad Ali won an Olympic gold medal in boxing before turning professional.",
        "A perfect game in bowling requires twelve consecutive strikes.",
        "Formula 1 cars are permitted to refuel during a race under current regulations.",
    ],
    "Food & Nutrition": [
        "Honey is produced by bees regurgitating and processing flower nectar.",
        "All wine improves in quality the longer it is aged, regardless of type.",
        "Dark chocolate generally contains more cocoa solids than milk chocolate.",
        "Caffeine content is always higher in coffee than in an equivalent serving of energy drinks.",
        "Kimchi is a traditionally fermented Korean side dish.",
        "Avocados are classified botanically as a single-seeded berry.",
        "Adding oil to pasta cooking water prevents the noodles from sticking together.",
        "Vanilla is the second most expensive spice in the world after saffron.",
        "Freezing food at any temperature instantly kills all foodborne bacteria present.",
        "Tomatoes are native to South America before being introduced to Europe.",
        "MSG is a naturally occurring compound found in foods like tomatoes and cheese.",
    ],
    "Arts, Culture & Literature": [
        "Leonardo da Vinci kept detailed notebooks written largely in mirror-image script.",
        "Charles Dickens originally published many of his novels in serialized magazine installments.",
        "The play \"Hamlet\" was written by William Shakespeare in the early 1600s.",
        "Pablo Picasso co-founded the Cubist art movement.",
        "\"Moby-Dick\" was a commercial bestseller immediately upon its original publication.",
        "The Louvre Museum is the most-visited art museum in the world.",
        "Ludwig van Beethoven composed exclusively during the Classical period, never the Romantic era.",
        "Jane Austen published \"Pride and Prejudice\" under her own name at its first printing.",
        "The Sistine Chapel ceiling took Michelangelo roughly four years to paint.",
        "Ernest Hemingway won the Nobel Prize in Literature during his lifetime.",
    ],
    "Law & Crime": [
        "The Miranda warning must be read to every person at the moment of arrest.",
        "Jury nullification allows a jury to acquit a defendant despite the evidence of guilt.",
        "The exclusionary rule prevents illegally obtained evidence from being used in a US criminal trial.",
        "Attorney-client privilege can be waived unintentionally in certain circumstances.",
        "The insanity defense is successfully used in the majority of US felony trials.",
        "Interpol maintains its own independent police force that can make arrests directly.",
        "Habeas corpus allows a detained person to challenge the legality of their detention.",
        "The statute of limitations never applies to any type of criminal charge.",
        "Double jeopardy protections in the US prevent retrial after an acquittal in the same jurisdiction.",
        "The Fifth Amendment protects individuals from being compelled to testify against themselves.",
    ],
}


def ordered_claims() -> List[Tuple[str, str]]:
    """Return [(domain, claim)] round-robin across domains — deterministic."""
    buckets = {d: list(c) for d, c in CLAIMS_BY_DOMAIN.items()}
    ordered: List[Tuple[str, str]] = []
    while any(buckets.values()):
        for domain in CLAIMS_BY_DOMAIN:
            if buckets[domain]:
                ordered.append((domain, buckets[domain].pop(0)))
    return ordered


TOTAL_CLAIMS = sum(len(v) for v in CLAIMS_BY_DOMAIN.values())


if __name__ == "__main__":
    ordered = ordered_claims()
    print(f"{TOTAL_CLAIMS} claims across {len(CLAIMS_BY_DOMAIN)} domains")
    assert len(ordered) == TOTAL_CLAIMS, "ordering dropped claims"
    assert len({c for _, c in ordered}) == TOTAL_CLAIMS, "duplicate claim text within v2"
    for d, items in CLAIMS_BY_DOMAIN.items():
        print(f"  {d}: {len(items)}")
