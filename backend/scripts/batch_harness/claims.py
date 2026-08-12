"""The 200-claim corpus for the batch test run.

Claims are grouped by domain here for readability, but `ordered_claims()`
returns them round-robin across domains. That ordering matters: NewsAPI's
free tier (100 req/day) is exhausted after roughly 30 claims, so whichever
claims run first are the only ones that see it. Interleaving spreads that
head across every domain instead of concentrating it in one topic.

Mix is roughly balanced between well-supported true claims, common false
claims/myths, and a few genuinely contested ones. No claim here was used in
earlier ad-hoc testing.
"""
from typing import Dict, List, Tuple

CLAIMS_BY_DOMAIN: Dict[str, List[str]] = {
    "Science — Physics & Chemistry": [
        "Sound travels faster in water than in air.",
        "Diamond is the hardest naturally occurring substance on Earth.",
        "Absolute zero is equal to negative 273.15 degrees Celsius.",
        "Helium is the second most abundant element in the universe.",
        "Rubber is an electrical conductor.",
        "The speed of light in a vacuum is approximately 300,000 kilometres per second.",
        "Water expands when it freezes.",
        "Gold is chemically inert and does not tarnish.",
        "Nuclear fusion powers hydrogen bombs.",
        "Sound cannot travel through a vacuum.",
        "Adding salt to water raises its boiling point.",
        "Graphene is a single layer of carbon atoms.",
        "Radioactive half-life can be changed by heating the material.",
        "Superconductors have exactly zero electrical resistance.",
        "Lead can be turned into gold by a chemical reaction.",
    ],
    "Science — Biology": [
        "Human DNA is about 99 percent identical between any two individuals.",
        "Sharks are immune to cancer.",
        "Octopuses have three hearts.",
        "Bananas are botanically classified as berries.",
        "Humans have exactly five senses.",
        "Bats are blind.",
        "A tomato is botanically a fruit.",
        "Mitochondria have their own DNA separate from the nucleus.",
        "Cows have four stomachs.",
        "Humans share about 50 percent of their DNA with bananas.",
        "Hair and fingernails continue to grow after death.",
        "Antibiotics are effective against viral infections.",
        "Honey never spoils.",
        "Butterflies remember experiences from their caterpillar stage.",
        "All mammals give birth to live young.",
    ],
    "Health & Medicine": [
        "Vitamin C prevents the common cold.",
        "Reading in dim light permanently damages your eyesight.",
        "Eating carrots significantly improves night vision.",
        "The flu vaccine can give you the flu.",
        "Drinking eight glasses of water a day is medically required.",
        "Penicillin was discovered by Alexander Fleming.",
        "Cold weather causes the common cold.",
        "MSG causes headaches in most people.",
        "Aspirin was originally derived from willow bark.",
        "Insulin was first isolated at the University of Toronto.",
        "Detox diets remove toxins from the body.",
        "Wearing a hat causes hair loss.",
        "Chemotherapy always causes complete hair loss.",
        "The human appendix serves no biological function.",
        "Blood in human veins is blue until it is exposed to oxygen.",
    ],
    "History": [
        "The Great Fire of London occurred in 1666.",
        "Vikings wore horned helmets in battle.",
        "Cleopatra lived closer in time to the Moon landing than to the building of the Great Pyramid.",
        "The Berlin Wall fell in 1989.",
        'Marie Antoinette said "let them eat cake" about starving peasants.',
        "The Hundred Years' War lasted exactly one hundred years.",
        "Julius Caesar was born by caesarean section.",
        "The Titanic sank on its maiden voyage in 1912.",
        "Christopher Columbus was the first European to reach the Americas.",
        "The Rosetta Stone was key to deciphering Egyptian hieroglyphs.",
        "Nero played the fiddle while Rome burned.",
        "The Magna Carta was sealed in 1215.",
        "Ancient Roman gladiators always fought to the death.",
        "The Salem witch trials resulted in people being burned at the stake.",
        "World War I ended on 11 November 1918.",
        "The Mongol Empire under Genghis Khan was the largest contiguous land empire in history.",
    ],
    "Politics & Government": [
        "The United Nations was founded in 1945.",
        "Switzerland is a member of the European Union.",
        "The US Constitution has been amended 27 times.",
        "Australia's capital city is Sydney.",
        "The UK Prime Minister is directly elected by voters.",
        "Nelson Mandela was imprisoned for 27 years.",
        "The European Union has 27 member states.",
        "Turkey is a member of NATO.",
        "A US President can serve unlimited terms.",
        "India is the world's largest democracy by population.",
        "Brazil's capital is Rio de Janeiro.",
        "The Kyoto Protocol was adopted in 1997.",
        "New Zealand was the first country to grant women the right to vote.",
        "The US Senate has 100 members, two from each state.",
        "Canada has no official languages.",
    ],
    "Economics & Finance": [
        "The 2008 financial crisis was triggered by subprime mortgage defaults.",
        "Hyperinflation in Zimbabwe led to a 100 trillion dollar banknote.",
        "The Federal Reserve is a private company owned by commercial banks.",
        "Gold prices always rise during a recession.",
        "The euro was introduced as physical currency in 2002.",
        "Japan has the world's largest economy by GDP.",
        "Inflation reduces the purchasing power of money.",
        "The Great Depression began with the 1929 stock market crash.",
        "Ethereum uses a proof-of-stake consensus mechanism.",
        "Raising the minimum wage always increases unemployment.",
        "The World Bank and the IMF are the same institution.",
        "Sweden is close to becoming a cashless society.",
        "Tulip mania caused a nationwide economic collapse in the Netherlands.",
        "Compound interest causes investments to grow exponentially over time.",
    ],
    "Geography & Travel": [
        "Africa is the largest continent by land area.",
        "The Dead Sea is the lowest point on Earth's land surface.",
        "Istanbul spans two continents.",
        "Greenland is larger than Africa.",
        "The Sahara is the largest desert in the world.",
        "Russia spans eleven time zones.",
        "Vatican City is the smallest sovereign state in the world.",
        "Mount Kilimanjaro is located in Kenya.",
        "The Nile flows northward.",
        "Iceland is largely covered in ice year-round.",
        "Bolivia has two capital cities.",
        "The Great Barrier Reef is visible from low Earth orbit.",
        "Lake Baikal holds about 20 percent of the world's unfrozen fresh water.",
        "Australia is both a country and a continent.",
        "Exactly seven continents are recognized worldwide.",
    ],
    "Space & Astronomy": [
        "Pluto was reclassified as a dwarf planet in 2006.",
        "Venus is the hottest planet in the solar system.",
        "Saturn is the only planet with rings.",
        "A day on Venus is longer than its year.",
        "The Milky Way will collide with the Andromeda galaxy.",
        "Black holes emit Hawking radiation.",
        "Jupiter has more than 90 known moons.",
        "The Moon has no gravity.",
        "Astronauts float in orbit because there is no gravity in space.",
        "Mars appears red due to iron oxide on its surface.",
        "The James Webb Space Telescope orbits the Earth directly.",
        "Neil Armstrong and Buzz Aldrin walked on the Moon in 1969.",
        "The Sun is classified as a yellow dwarf star.",
        "Light from the Sun takes about eight minutes to reach Earth.",
        "Voyager 1 has left the solar system entirely.",
    ],
    "Computer Science & Technology": [
        "The first computer bug was an actual moth found in a relay.",
        "Python was created by Guido van Rossum.",
        "HTTP is a stateless protocol.",
        "Quantum computers can solve all NP-complete problems in polynomial time.",
        "Linux uses a monolithic kernel.",
        "Bitcoin's blockchain processes more transactions per second than Visa.",
        "RSA encryption relies on the difficulty of factoring large numbers.",
        "Ada Lovelace wrote the first computer algorithm.",
        "Moore's Law states that transistor counts double roughly every two years.",
        "HTML is a programming language.",
        "The internet and the World Wide Web are the same thing.",
        "Alan Turing helped break the Enigma cipher during World War II.",
        "Git was created by Linus Torvalds.",
        "Deleting a file from a hard drive immediately erases the underlying data.",
        "JavaScript and Java are the same language.",
        "The first email was sent in 1971.",
        "P equals NP has been proven.",
        "Large language models are trained using gradient descent.",
    ],
    "Social Media & Online Controversy": [
        "Facebook was originally called The Facebook.",
        "Twitter was rebranded to X in 2023.",
        "TikTok is owned by the Chinese company ByteDance.",
        "The Cambridge Analytica scandal involved harvested Facebook user data.",
        "Instagram was acquired by Facebook for one billion dollars.",
        "YouTube was originally created as a dating website.",
        "Elon Musk purchased Twitter for 44 billion dollars.",
        "Deleting a social media account immediately erases all your data from company servers.",
        "WhatsApp messages are protected by end-to-end encryption.",
        "The Momo Challenge caused a documented wave of child suicides.",
        "Social media algorithms are legally required to be transparent in the United States.",
        "MySpace was the most visited website in the United States in 2006.",
        "Reddit was founded in 2005.",
        "Instagram hid public like counts globally in 2019.",
        "LinkedIn is owned by Microsoft.",
    ],
    "Environment & Climate": [
        "Atmospheric carbon dioxide has exceeded 400 parts per million.",
        "The ozone hole over Antarctica has been shrinking.",
        "Recycling plastic is always more energy efficient than producing new plastic.",
        "The Amazon rainforest produces 20 percent of the world's oxygen.",
        "Nuclear power plants emit large amounts of carbon dioxide during operation.",
        "Global sea levels have risen over the past century.",
        "Electric vehicles produce zero emissions over their entire lifecycle.",
        "The Paris Agreement was adopted in 2015.",
        "Global average temperatures have risen more than one degree Celsius since pre-industrial times.",
        "Wind turbines kill more birds than domestic cats do.",
        "Antarctica is gaining ice overall.",
        "Deforestation contributes to global carbon emissions.",
    ],
    "Sports": [
        "The modern Olympic Games began in 1896.",
        "Cristiano Ronaldo has won five Ballon d'Or awards.",
        "A marathon is exactly 42.195 kilometres long.",
        "The New Zealand All Blacks perform the haka before matches.",
        "Basketball was invented by James Naismith.",
        "The Tour de France is always held entirely within France.",
        "Cricket test matches can last up to five days.",
        "Michael Phelps has won more Olympic medals than any other athlete.",
        "Association football matches consist of two 45-minute halves.",
        "The Super Bowl is watched by more people worldwide than the FIFA World Cup final.",
    ],
    "Food & Nutrition": [
        "Carrots were originally purple before orange varieties were cultivated.",
        "Eating late at night directly causes weight gain regardless of total calories.",
        "Microwaving food destroys more nutrients than other cooking methods.",
        "Brown eggs are more nutritious than white eggs.",
        "Gluten is harmful to everyone, not just people with celiac disease.",
        "Alcohol directly kills brain cells.",
        "Spicy food causes stomach ulcers.",
        "Chocolate is toxic to dogs.",
        "Fresh vegetables are always more nutritious than frozen ones.",
        "Searing meat seals in its juices.",
    ],
    "Arts, Culture & Literature": [
        "Vincent van Gogh sold only one painting during his lifetime.",
        "William Shakespeare wrote 37 plays.",
        "The Mona Lisa was stolen from the Louvre in 1911.",
        "Mozart died while composing his Requiem.",
        "The novel Frankenstein was written by Mary Shelley.",
        "Beethoven was completely deaf when he composed his Ninth Symphony.",
        "The Great Gatsby was a bestseller when first published.",
        "Van Gogh cut off his entire ear.",
        "The Sistine Chapel ceiling was painted by Michelangelo.",
    ],
    "Law & Crime": [
        "Miranda rights originate from a 1966 US Supreme Court case.",
        "In the UK a person can never be tried twice for the same crime.",
        "The FBI was established in 1908.",
        "Fingerprints are unique to every individual, including identical twins.",
        "Interpol has the authority to arrest suspects in member countries.",
        "DNA evidence has exonerated hundreds of wrongfully convicted people in the United States.",
    ],
}


def ordered_claims() -> List[Tuple[str, str]]:
    """Return [(domain, claim)] round-robin across domains.

    Deterministic — the same order on every run, so `claim_id`s and resume
    behaviour stay stable across restarts.
    """
    buckets = {d: list(c) for d, c in CLAIMS_BY_DOMAIN.items()}
    ordered: List[Tuple[str, str]] = []
    while any(buckets.values()):
        for domain in CLAIMS_BY_DOMAIN:          # dict preserves insertion order
            if buckets[domain]:
                ordered.append((domain, buckets[domain].pop(0)))
    return ordered


TOTAL_CLAIMS = sum(len(v) for v in CLAIMS_BY_DOMAIN.values())


if __name__ == "__main__":
    ordered = ordered_claims()
    print(f"{TOTAL_CLAIMS} claims across {len(CLAIMS_BY_DOMAIN)} domains")
    assert len(ordered) == TOTAL_CLAIMS, "ordering dropped claims"
    assert len({c for _, c in ordered}) == TOTAL_CLAIMS, "duplicate claim text"
    print("\nFirst 15 (the NewsAPI-rich head):")
    for i, (domain, claim) in enumerate(ordered[:15], 1):
        print(f"  {i:3d}. [{domain}] {claim[:62]}")
