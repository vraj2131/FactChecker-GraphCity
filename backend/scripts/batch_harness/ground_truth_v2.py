"""Reference answers for the claims_v2.py corpus (the second 200-claim run).

Same rubric as ground_truth.py: TRUE / FALSE / CONTESTED, independent of
what the pipeline returns. See that module's docstring for the full
methodology and its caveats (self-graded, not independently audited).
"""
from typing import Dict

GROUND_TRUTH: Dict[str, str] = {
    # ── Common Myths & Misconceptions ───────────────────────────────────────
    "Toads or frogs give you warts if you touch them.": "FALSE",
    "Poinsettia plants are highly poisonous if a person or pet eats a leaf.": "FALSE",
    "Police require a 24-hour waiting period before they can investigate a missing person.": "FALSE",
    "Urinating on a jellyfish sting relieves the pain.": "FALSE",
    "The Coriolis effect determines which direction water drains in a sink or toilet.": "FALSE",
    "People are strictly left-brained or right-brained depending on whether they are more logical or creative.": "FALSE",
    "A duck's quack does not produce an echo.": "FALSE",
    "Camels store water in their humps.": "FALSE",
    "Standing near a running microwave oven exposes you to harmful radiation leakage.": "FALSE",
    "Fortune cookies originated in ancient China.": "FALSE",
    "Fish do not feel pain the way mammals and birds do.": "FALSE",
    "Fingernails need to \"breathe\" and are damaged by wearing nail polish for long periods.": "FALSE",
    "An electric fan running overnight in a sealed room can cause a person to suffocate.": "FALSE",
    "Planting marigolds around a garden repels all harmful insects.": "FALSE",
    "Poison ivy rashes can spread across the body by scratching the affected skin.": "FALSE",
    "Swimming immediately after eating a meal causes severe cramps and drowning.": "FALSE",
    "The far side of the Moon never receives any sunlight.": "FALSE",
    "Humans are born with an instinctive fear of snakes and spiders.": "CONTESTED",
    "A patient's body can develop antibiotic resistance during a single course of treatment.": "FALSE",
    "An adult's core personality traits are permanently fixed and cannot meaningfully change.": "FALSE",
    "Chewing celery burns more calories than the celery itself contains.": "FALSE",
    "The Vikings were the first Europeans to reach North America.": "TRUE",
    "Lemmings will follow each other off cliffs in mass suicidal migrations.": "FALSE",
    "Different species can never interbreed to produce fertile offspring under any circumstances.": "FALSE",
    "Touching a baby bird will cause its mother to abandon it because of the human scent.": "FALSE",
    "Static electric shocks in winter are caused by dry air alone, regardless of clothing material.": "FALSE",
    "The five-second rule makes food safe to eat if picked up quickly enough after being dropped.": "FALSE",
    "Earwigs commonly crawl into people's ears while they sleep.": "FALSE",
    "Humans and dinosaurs coexisted at the same time on Earth.": "FALSE",
    "Cold showers are more effective than vaccines at boosting the immune system.": "FALSE",
    "Drinking alcohol effectively warms up the body during cold weather.": "FALSE",
    "Ancient Egyptians used slave labor exclusively to build the pyramids.": "FALSE",
    "Sharks must keep swimming constantly or they will suffocate and die.": "FALSE",
    "A person's IQ score is fixed at birth and cannot change throughout life.": "FALSE",
    "Peanuts are classified botanically as nuts.": "FALSE",
    "The human tongue is the strongest muscle in the body relative to its size.": "CONTESTED",
    "Ostrich chicks can outrun a galloping horse from the moment they hatch.": "FALSE",
    "A goldfish kept in a small bowl will only grow as large as its container allows.": "FALSE",
    "Swallowing watermelon seeds will cause a watermelon to grow inside your stomach.": "FALSE",
    "The Sahara Desert was always a vast expanse of sand dunes throughout ancient history.": "FALSE",

    # ── Science — Physics & Chemistry ───────────────────────────────────────
    "Diamonds are formed exclusively from compressed coal deep underground.": "FALSE",
    "Helium makes your voice sound higher because it is denser than air.": "FALSE",
    "Glass shatters more easily at extremely low temperatures than at room temperature.": "TRUE",
    "Mixing bleach and ammonia produces a toxic gas.": "TRUE",
    "The freezing point of pure water at sea level is exactly 0 degrees Celsius.": "TRUE",
    "Static electricity can be safely ignored around fuel pumps.": "FALSE",
    "Aluminum is more abundant in Earth's crust than iron.": "TRUE",
    "Neon signs are filled exclusively with neon gas regardless of their color.": "FALSE",
    "A vacuum completely blocks the transfer of heat by all methods.": "FALSE",
    "Table salt is a compound made of sodium and chlorine atoms.": "TRUE",
    "Copper is a better conductor of electricity than silver.": "FALSE",

    # ── Science — Biology ────────────────────────────────────────────────────
    "Octopuses are capable of changing the texture of their skin, not just its color.": "TRUE",
    "The human body contains more bacterial cells than human cells.": "CONTESTED",
    "Elephants are the only mammals that cannot jump.": "CONTESTED",
    "Koalas sleep for around 20 hours a day.": "TRUE",
    "Human tooth enamel is the hardest substance in the human body.": "TRUE",
    "All spiders produce venom that is dangerous to humans.": "FALSE",
    "Sea otters use rocks as tools to crack open shellfish.": "TRUE",
    "Human red blood cells lack a nucleus when mature.": "TRUE",
    "Kangaroos cannot walk backwards.": "TRUE",
    "The liver is the only human organ capable of regenerating lost tissue.": "FALSE",
    "Praying mantises always eat their mate's head during mating.": "FALSE",

    # ── Health & Medicine ────────────────────────────────────────────────────
    "Ibuprofen and acetaminophen work through the same biological mechanism.": "FALSE",
    "The human body's core temperature is exactly 98.6 degrees Fahrenheit for everyone.": "FALSE",
    "Insulin was the first hormone to be chemically synthesized in a laboratory.": "FALSE",
    "Smallpox is the only human disease to have been fully eradicated worldwide.": "TRUE",
    "Type 1 diabetes is caused primarily by diet and lifestyle choices.": "FALSE",
    "The BCG vaccine was developed primarily to prevent tuberculosis.": "TRUE",
    "Human blood types are determined by antigens present on red blood cells.": "TRUE",
    "Morphine is derived from the opium poppy plant.": "TRUE",
    "The common cold and influenza are caused by the same family of viruses.": "FALSE",
    "Regular exercise has been shown to reduce the risk of several types of cancer.": "TRUE",
    "X-rays were discovered by Wilhelm Conrad Röntgen.": "TRUE",

    # ── History ──────────────────────────────────────────────────────────────
    "The printing press was invented by Johannes Gutenberg in the 15th century.": "TRUE",
    "The Roman Empire officially split into Western and Eastern halves in 285 CE under Diocletian's reforms.": "CONTESTED",
    "The Wright brothers' first powered flight took place in 1903.": "TRUE",
    "The Suez Canal was completed and opened in 1869.": "TRUE",
    "King Tutankhamun's tomb was discovered by Howard Carter in 1922.": "TRUE",
    "The Boston Tea Party took place in 1773.": "TRUE",
    "The Ottoman Empire lasted for more than 600 years.": "TRUE",
    "Anne Frank's diary was published after being found by her father.": "TRUE",
    "The first modern Olympic marathon distance was standardized at the 1908 London Games.": "TRUE",
    "The Black Death killed the majority of Europe's population in the 14th century.": "TRUE",
    "Ancient Sparta was governed by a single, all-powerful king.": "FALSE",

    # ── Politics & Government ───────────────────────────────────────────────
    "The United States has a bicameral national legislature.": "TRUE",
    "Germany's head of government is called the Chancellor.": "TRUE",
    "The International Criminal Court has the power to prosecute any country's leader without exception.": "FALSE",
    "South Africa held its first fully democratic elections in 1994.": "TRUE",
    "The Vatican is the smallest independent state to hold a seat at the United Nations General Assembly.": "FALSE",
    "Japan's constitution renounces the right to maintain offensive military forces.": "TRUE",
    "The European Parliament is directly elected by citizens of EU member states.": "TRUE",
    "Every member of the United Nations Security Council holds veto power.": "FALSE",
    "Iceland has not maintained a standing army for decades.": "TRUE",
    "The president of France is elected to a five-year term.": "TRUE",
    "All fifty US states require a majority popular vote to award their electoral votes.": "FALSE",

    # ── Economics & Finance ──────────────────────────────────────────────────
    "The London Stock Exchange is one of the oldest stock exchanges in the world.": "TRUE",
    "Cryptocurrency transactions are always completely anonymous and untraceable.": "FALSE",
    "The gold standard was fully abandoned by the United States in 1971.": "TRUE",
    "Venture capital funding is a primary source of financing for most small businesses globally.": "FALSE",
    "The Consumer Price Index is commonly used to measure inflation.": "TRUE",
    "Central banks can never lose money by printing more currency.": "FALSE",
    "The Bretton Woods Agreement established the International Monetary Fund.": "TRUE",
    "Diversifying an investment portfolio eliminates all financial risk.": "FALSE",
    "Japan experienced a prolonged period of economic stagnation often called the \"Lost Decade.\"": "TRUE",
    "Credit scores are calculated identically by every credit bureau.": "FALSE",
    "Adam Smith is widely regarded as a founding figure of modern economics.": "TRUE",

    # ── Geography & Travel ───────────────────────────────────────────────────
    "Russia is the largest country in the world by total land area.": "TRUE",
    "The Amazon River carries more water than any other river on Earth.": "TRUE",
    "Canada has more freshwater lakes than any other country.": "TRUE",
    "Mount Everest's summit is the point farthest from the Earth's core.": "FALSE",
    "The Panama Canal connects the Atlantic and Pacific Oceans.": "TRUE",
    "Monaco is the second-smallest country in the world by land area.": "TRUE",
    "New Zealand is located directly southeast of Australia.": "TRUE",
    "The Amazon rainforest lies entirely within Brazil's borders.": "FALSE",
    "Chile is the longest country in the world from north to south.": "CONTESTED",
    "The Strait of Gibraltar separates Europe from Africa.": "TRUE",
    "Death Valley holds the record for the highest air temperature ever reliably recorded on Earth.": "TRUE",

    # ── Space & Astronomy ────────────────────────────────────────────────────
    "Mercury is the closest planet to the Sun.": "TRUE",
    "A shooting star is typically a small piece of interplanetary debris burning up in the atmosphere.": "TRUE",
    "The International Space Station orbits Earth roughly every 90 minutes.": "TRUE",
    "Uranus rotates on an axis that is tilted almost sideways relative to its orbit.": "TRUE",
    "Betelgeuse is expected to eventually explode as a supernova.": "TRUE",
    "The Hubble Space Telescope was launched in 1990.": "TRUE",
    "Mars has two small moons named Phobos and Deimos.": "TRUE",
    "Solar eclipses can only occur during a new moon phase.": "TRUE",
    "The asteroid belt lies between the orbits of Mars and Jupiter.": "TRUE",
    "A light-year is a measurement of time rather than distance.": "FALSE",
    "Saturn's density is low enough that it would float in water.": "TRUE",

    # ── Computer Science & Technology ────────────────────────────────────────
    "TCP guarantees reliable, ordered delivery of data packets across a network.": "TRUE",
    "The first practical web browser was called Mosaic.": "TRUE",
    "Binary code represents data using only the digits 0 and 1.": "TRUE",
    "A compiler and an interpreter perform the exact same function in software execution.": "FALSE",
    "The C programming language was developed at Bell Labs.": "TRUE",
    "Public-key cryptography requires both parties to share the same secret key in advance.": "FALSE",
    "USB flash drives use magnetic storage to retain data.": "FALSE",
    "Assembly language provides a direct one-to-one mapping to a specific processor's machine instructions.": "TRUE",
    "Wikipedia is edited exclusively by a small paid staff of professional writers.": "FALSE",
    "Open-source software can never be sold commercially.": "FALSE",
    "The Turing test was proposed as a way to evaluate machine intelligence.": "TRUE",

    # ── Social Media & Online Controversy ───────────────────────────────────
    "Snapchat pioneered the disappearing-message format before other major platforms adopted it.": "TRUE",
    "Pinterest was originally launched as an invite-only platform.": "TRUE",
    "The first tweet ever posted was sent by Twitter co-founder Jack Dorsey.": "TRUE",
    "Discord was originally created primarily as a tool for gamers to communicate.": "TRUE",
    "Wikipedia's content is licensed under a copyleft license that permits reuse.": "TRUE",
    "Clubhouse's audio-only format required an iPhone at its initial launch.": "TRUE",
    "Vine was shut down entirely before TikTok's short-video format gained popularity.": "TRUE",
    "All content posted publicly on social media automatically enters the public domain.": "FALSE",
    "YouTube's algorithm relies solely on view counts to rank recommended videos.": "FALSE",
    "Telegram offers end-to-end encryption by default in all standard one-on-one chats.": "FALSE",

    # ── Environment & Climate ────────────────────────────────────────────────
    "Coral bleaching is primarily caused by rising ocean water temperatures.": "TRUE",
    "Methane is a more potent greenhouse gas than carbon dioxide over a 20-year period.": "TRUE",
    "The Kyoto Protocol legally bound all United Nations member countries to emissions targets.": "FALSE",
    "Deforestation rates in the Amazon have remained completely constant since 1970.": "FALSE",
    "Renewable energy sources like solar and wind cannot be stored using any current battery technology.": "FALSE",
    "Permafrost thaw can release stored greenhouse gases into the atmosphere.": "TRUE",
    "The hole in the ozone layer was primarily caused by chlorofluorocarbon emissions.": "TRUE",
    "Composting food waste produces zero methane emissions.": "FALSE",
    "Global plastic production has increased significantly since the mid-20th century.": "TRUE",
    "Fracking has no documented connection to increased seismic activity in any region.": "FALSE",

    # ── Sports ───────────────────────────────────────────────────────────────
    "The Iditarod is a long-distance sled dog race held in Alaska.": "TRUE",
    "A regulation FIFA soccer match lasts 90 minutes of play before stoppage time.": "TRUE",
    "Table tennis originated in England as an indoor version of lawn tennis.": "TRUE",
    "The Ryder Cup is a golf competition between teams from the United States and Europe.": "TRUE",
    "Wimbledon is the oldest tennis tournament in the world.": "TRUE",
    "Ice hockey periods in the NHL are each 20 minutes long.": "TRUE",
    "The Boston Marathon's course is certified downhill, making all its times ineligible for world records.": "FALSE",
    "Muhammad Ali won an Olympic gold medal in boxing before turning professional.": "TRUE",
    "A perfect game in bowling requires twelve consecutive strikes.": "TRUE",
    "Formula 1 cars are permitted to refuel during a race under current regulations.": "FALSE",

    # ── Food & Nutrition ─────────────────────────────────────────────────────
    "Honey is produced by bees regurgitating and processing flower nectar.": "TRUE",
    "All wine improves in quality the longer it is aged, regardless of type.": "FALSE",
    "Dark chocolate generally contains more cocoa solids than milk chocolate.": "TRUE",
    "Caffeine content is always higher in coffee than in an equivalent serving of energy drinks.": "FALSE",
    "Kimchi is a traditionally fermented Korean side dish.": "TRUE",
    "Avocados are classified botanically as a single-seeded berry.": "TRUE",
    "Adding oil to pasta cooking water prevents the noodles from sticking together.": "FALSE",
    "Vanilla is the second most expensive spice in the world after saffron.": "TRUE",
    "Freezing food at any temperature instantly kills all foodborne bacteria present.": "FALSE",
    "Tomatoes are native to South America before being introduced to Europe.": "TRUE",
    "MSG is a naturally occurring compound found in foods like tomatoes and cheese.": "TRUE",

    # ── Arts, Culture & Literature ───────────────────────────────────────────
    "Leonardo da Vinci kept detailed notebooks written largely in mirror-image script.": "TRUE",
    "Charles Dickens originally published many of his novels in serialized magazine installments.": "TRUE",
    "The play \"Hamlet\" was written by William Shakespeare in the early 1600s.": "TRUE",
    "Pablo Picasso co-founded the Cubist art movement.": "TRUE",
    "\"Moby-Dick\" was a commercial bestseller immediately upon its original publication.": "FALSE",
    "The Louvre Museum is the most-visited art museum in the world.": "TRUE",
    "Ludwig van Beethoven composed exclusively during the Classical period, never the Romantic era.": "FALSE",
    "Jane Austen published \"Pride and Prejudice\" under her own name at its first printing.": "FALSE",
    "The Sistine Chapel ceiling took Michelangelo roughly four years to paint.": "TRUE",
    "Ernest Hemingway won the Nobel Prize in Literature during his lifetime.": "TRUE",

    # ── Law & Crime ──────────────────────────────────────────────────────────
    "The Miranda warning must be read to every person at the moment of arrest.": "FALSE",
    "Jury nullification allows a jury to acquit a defendant despite the evidence of guilt.": "TRUE",
    "The exclusionary rule prevents illegally obtained evidence from being used in a US criminal trial.": "TRUE",
    "Attorney-client privilege can be waived unintentionally in certain circumstances.": "TRUE",
    "The insanity defense is successfully used in the majority of US felony trials.": "FALSE",
    "Interpol maintains its own independent police force that can make arrests directly.": "FALSE",
    "Habeas corpus allows a detained person to challenge the legality of their detention.": "TRUE",
    "The statute of limitations never applies to any type of criminal charge.": "FALSE",
    "Double jeopardy protections in the US prevent retrial after an acquittal in the same jurisdiction.": "TRUE",
    "The Fifth Amendment protects individuals from being compelled to testify against themselves.": "TRUE",
}

EXPECTED_VERDICT = {"TRUE": "verified", "FALSE": "rejected"}


def score(label: str, verdict: str) -> str:
    """Identical rubric to ground_truth.score() — see that module for docs."""
    if label == "CONTESTED":
        return "contested_abstained" if verdict == "not_enough_info" else "contested_decided"
    expected = EXPECTED_VERDICT[label]
    if verdict == expected:
        return "correct"
    if verdict == "not_enough_info":
        return "missed"
    return "wrong"
