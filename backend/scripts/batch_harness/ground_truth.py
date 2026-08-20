"""Reference answers for the 200-claim corpus.

Labels are assigned from the claim's factual status independent of what the
pipeline returned, so they can be used to score it. Keyed by claim text
rather than index so reordering claims.py cannot silently misalign them.

  TRUE       — well established; the correct verdict is "verified"
  FALSE      — well established as false; correct verdict is "rejected"
  CONTESTED  — genuinely disputed, definition-dependent, or true only under a
               qualification the claim omits. "not_enough_info" is a defensible
               answer here, so these are scored separately rather than counted
               as errors either way.
"""
from typing import Dict

GROUND_TRUTH: Dict[str, str] = {
    # ── Science — Physics & Chemistry ──────────────────────────────────────
    "Sound travels faster in water than in air.": "TRUE",
    "Diamond is the hardest naturally occurring substance on Earth.": "TRUE",
    "Absolute zero is equal to negative 273.15 degrees Celsius.": "TRUE",
    "Helium is the second most abundant element in the universe.": "TRUE",
    "Rubber is an electrical conductor.": "FALSE",
    "The speed of light in a vacuum is approximately 300,000 kilometres per second.": "TRUE",
    "Water expands when it freezes.": "TRUE",
    "Gold is chemically inert and does not tarnish.": "TRUE",
    "Nuclear fusion powers hydrogen bombs.": "TRUE",
    "Sound cannot travel through a vacuum.": "TRUE",
    "Adding salt to water raises its boiling point.": "TRUE",
    "Graphene is a single layer of carbon atoms.": "TRUE",
    "Radioactive half-life can be changed by heating the material.": "FALSE",
    "Superconductors have exactly zero electrical resistance.": "TRUE",
    "Lead can be turned into gold by a chemical reaction.": "FALSE",

    # ── Science — Biology ──────────────────────────────────────────────────
    "Human DNA is about 99 percent identical between any two individuals.": "TRUE",
    "Sharks are immune to cancer.": "FALSE",
    "Octopuses have three hearts.": "TRUE",
    "Bananas are botanically classified as berries.": "TRUE",
    "Humans have exactly five senses.": "FALSE",
    "Bats are blind.": "FALSE",
    "A tomato is botanically a fruit.": "TRUE",
    "Mitochondria have their own DNA separate from the nucleus.": "TRUE",
    "Cows have four stomachs.": "CONTESTED",   # one stomach, four compartments
    "Humans share about 50 percent of their DNA with bananas.": "TRUE",
    "Hair and fingernails continue to grow after death.": "FALSE",
    "Antibiotics are effective against viral infections.": "FALSE",
    "Honey never spoils.": "TRUE",
    "Butterflies remember experiences from their caterpillar stage.": "TRUE",
    "All mammals give birth to live young.": "FALSE",

    # ── Health & Medicine ──────────────────────────────────────────────────
    "Vitamin C prevents the common cold.": "FALSE",
    "Reading in dim light permanently damages your eyesight.": "FALSE",
    "Eating carrots significantly improves night vision.": "FALSE",
    "The flu vaccine can give you the flu.": "FALSE",
    "Drinking eight glasses of water a day is medically required.": "FALSE",
    "Penicillin was discovered by Alexander Fleming.": "TRUE",
    "Cold weather causes the common cold.": "FALSE",
    "MSG causes headaches in most people.": "FALSE",
    "Aspirin was originally derived from willow bark.": "TRUE",
    "Insulin was first isolated at the University of Toronto.": "TRUE",
    "Detox diets remove toxins from the body.": "FALSE",
    "Wearing a hat causes hair loss.": "FALSE",
    "Chemotherapy always causes complete hair loss.": "FALSE",
    "The human appendix serves no biological function.": "FALSE",
    "Blood in human veins is blue until it is exposed to oxygen.": "FALSE",

    # ── History ────────────────────────────────────────────────────────────
    "The Great Fire of London occurred in 1666.": "TRUE",
    "Vikings wore horned helmets in battle.": "FALSE",
    "Cleopatra lived closer in time to the Moon landing than to the building of the Great Pyramid.": "TRUE",
    "The Berlin Wall fell in 1989.": "TRUE",
    'Marie Antoinette said "let them eat cake" about starving peasants.': "FALSE",
    "The Hundred Years' War lasted exactly one hundred years.": "FALSE",
    "Julius Caesar was born by caesarean section.": "FALSE",
    "The Titanic sank on its maiden voyage in 1912.": "TRUE",
    "Christopher Columbus was the first European to reach the Americas.": "FALSE",
    "The Rosetta Stone was key to deciphering Egyptian hieroglyphs.": "TRUE",
    "Nero played the fiddle while Rome burned.": "FALSE",
    "The Magna Carta was sealed in 1215.": "TRUE",
    "Ancient Roman gladiators always fought to the death.": "FALSE",
    "The Salem witch trials resulted in people being burned at the stake.": "FALSE",
    "World War I ended on 11 November 1918.": "TRUE",
    "The Mongol Empire under Genghis Khan was the largest contiguous land empire in history.": "TRUE",

    # ── Politics & Government ──────────────────────────────────────────────
    "The United Nations was founded in 1945.": "TRUE",
    "Switzerland is a member of the European Union.": "FALSE",
    "The US Constitution has been amended 27 times.": "TRUE",
    "Australia's capital city is Sydney.": "FALSE",
    "The UK Prime Minister is directly elected by voters.": "FALSE",
    "Nelson Mandela was imprisoned for 27 years.": "TRUE",
    "The European Union has 27 member states.": "TRUE",
    "Turkey is a member of NATO.": "TRUE",
    "A US President can serve unlimited terms.": "FALSE",
    "India is the world's largest democracy by population.": "TRUE",
    "Brazil's capital is Rio de Janeiro.": "FALSE",
    "The Kyoto Protocol was adopted in 1997.": "TRUE",
    "New Zealand was the first country to grant women the right to vote.": "TRUE",
    "The US Senate has 100 members, two from each state.": "TRUE",
    "Canada has no official languages.": "FALSE",

    # ── Economics & Finance ────────────────────────────────────────────────
    "The 2008 financial crisis was triggered by subprime mortgage defaults.": "TRUE",
    "Hyperinflation in Zimbabwe led to a 100 trillion dollar banknote.": "TRUE",
    "The Federal Reserve is a private company owned by commercial banks.": "FALSE",
    "Gold prices always rise during a recession.": "FALSE",
    "The euro was introduced as physical currency in 2002.": "TRUE",
    "Japan has the world's largest economy by GDP.": "FALSE",
    "Inflation reduces the purchasing power of money.": "TRUE",
    "The Great Depression began with the 1929 stock market crash.": "TRUE",
    "Ethereum uses a proof-of-stake consensus mechanism.": "TRUE",
    "Raising the minimum wage always increases unemployment.": "FALSE",
    "The World Bank and the IMF are the same institution.": "FALSE",
    "Sweden is close to becoming a cashless society.": "TRUE",
    "Tulip mania caused a nationwide economic collapse in the Netherlands.": "FALSE",
    "Compound interest causes investments to grow exponentially over time.": "TRUE",

    # ── Geography & Travel ─────────────────────────────────────────────────
    "Africa is the largest continent by land area.": "FALSE",
    "The Dead Sea is the lowest point on Earth's land surface.": "TRUE",
    "Istanbul spans two continents.": "TRUE",
    "Greenland is larger than Africa.": "FALSE",
    "The Sahara is the largest desert in the world.": "CONTESTED",  # largest hot desert; Antarctica larger
    "Russia spans eleven time zones.": "TRUE",
    "Vatican City is the smallest sovereign state in the world.": "TRUE",
    "Mount Kilimanjaro is located in Kenya.": "FALSE",
    "The Nile flows northward.": "TRUE",
    "Iceland is largely covered in ice year-round.": "FALSE",
    "Bolivia has two capital cities.": "TRUE",
    "The Great Barrier Reef is visible from low Earth orbit.": "TRUE",
    "Lake Baikal holds about 20 percent of the world's unfrozen fresh water.": "TRUE",
    "Australia is both a country and a continent.": "TRUE",
    "Exactly seven continents are recognized worldwide.": "FALSE",

    # ── Space & Astronomy ──────────────────────────────────────────────────
    "Pluto was reclassified as a dwarf planet in 2006.": "TRUE",
    "Venus is the hottest planet in the solar system.": "TRUE",
    "Saturn is the only planet with rings.": "FALSE",
    "A day on Venus is longer than its year.": "TRUE",
    "The Milky Way will collide with the Andromeda galaxy.": "TRUE",
    "Black holes emit Hawking radiation.": "CONTESTED",  # theoretical, never observed
    "Jupiter has more than 90 known moons.": "TRUE",
    "The Moon has no gravity.": "FALSE",
    "Astronauts float in orbit because there is no gravity in space.": "FALSE",
    "Mars appears red due to iron oxide on its surface.": "TRUE",
    "The James Webb Space Telescope orbits the Earth directly.": "FALSE",
    "Neil Armstrong and Buzz Aldrin walked on the Moon in 1969.": "TRUE",
    "The Sun is classified as a yellow dwarf star.": "TRUE",
    "Light from the Sun takes about eight minutes to reach Earth.": "TRUE",
    "Voyager 1 has left the solar system entirely.": "FALSE",

    # ── Computer Science & Technology ──────────────────────────────────────
    "The first computer bug was an actual moth found in a relay.": "TRUE",
    "Python was created by Guido van Rossum.": "TRUE",
    "HTTP is a stateless protocol.": "TRUE",
    "Quantum computers can solve all NP-complete problems in polynomial time.": "FALSE",
    "Linux uses a monolithic kernel.": "TRUE",
    "Bitcoin's blockchain processes more transactions per second than Visa.": "FALSE",
    "RSA encryption relies on the difficulty of factoring large numbers.": "TRUE",
    "Ada Lovelace wrote the first computer algorithm.": "TRUE",
    "Moore's Law states that transistor counts double roughly every two years.": "TRUE",
    "HTML is a programming language.": "FALSE",
    "The internet and the World Wide Web are the same thing.": "FALSE",
    "Alan Turing helped break the Enigma cipher during World War II.": "TRUE",
    "Git was created by Linus Torvalds.": "TRUE",
    "Deleting a file from a hard drive immediately erases the underlying data.": "FALSE",
    "JavaScript and Java are the same language.": "FALSE",
    "The first email was sent in 1971.": "TRUE",
    "P equals NP has been proven.": "FALSE",
    "Large language models are trained using gradient descent.": "TRUE",

    # ── Social Media & Online Controversy ──────────────────────────────────
    "Facebook was originally called The Facebook.": "TRUE",
    "Twitter was rebranded to X in 2023.": "TRUE",
    "TikTok is owned by the Chinese company ByteDance.": "TRUE",
    "The Cambridge Analytica scandal involved harvested Facebook user data.": "TRUE",
    "Instagram was acquired by Facebook for one billion dollars.": "TRUE",
    "YouTube was originally created as a dating website.": "TRUE",
    "Elon Musk purchased Twitter for 44 billion dollars.": "TRUE",
    "Deleting a social media account immediately erases all your data from company servers.": "FALSE",
    "WhatsApp messages are protected by end-to-end encryption.": "TRUE",
    "The Momo Challenge caused a documented wave of child suicides.": "FALSE",
    "Social media algorithms are legally required to be transparent in the United States.": "FALSE",
    "MySpace was the most visited website in the United States in 2006.": "TRUE",
    "Reddit was founded in 2005.": "TRUE",
    "Instagram hid public like counts globally in 2019.": "FALSE",
    "LinkedIn is owned by Microsoft.": "TRUE",

    # ── Environment & Climate ──────────────────────────────────────────────
    "Atmospheric carbon dioxide has exceeded 400 parts per million.": "TRUE",
    "The ozone hole over Antarctica has been shrinking.": "TRUE",
    "Recycling plastic is always more energy efficient than producing new plastic.": "FALSE",
    "The Amazon rainforest produces 20 percent of the world's oxygen.": "FALSE",
    "Nuclear power plants emit large amounts of carbon dioxide during operation.": "FALSE",
    "Global sea levels have risen over the past century.": "TRUE",
    "Electric vehicles produce zero emissions over their entire lifecycle.": "FALSE",
    "The Paris Agreement was adopted in 2015.": "TRUE",
    "Global average temperatures have risen more than one degree Celsius since pre-industrial times.": "TRUE",
    "Wind turbines kill more birds than domestic cats do.": "FALSE",
    "Antarctica is gaining ice overall.": "FALSE",
    "Deforestation contributes to global carbon emissions.": "TRUE",

    # ── Sports ─────────────────────────────────────────────────────────────
    "The modern Olympic Games began in 1896.": "TRUE",
    "Cristiano Ronaldo has won five Ballon d'Or awards.": "TRUE",
    "A marathon is exactly 42.195 kilometres long.": "TRUE",
    "The New Zealand All Blacks perform the haka before matches.": "TRUE",
    "Basketball was invented by James Naismith.": "TRUE",
    "The Tour de France is always held entirely within France.": "FALSE",
    "Cricket test matches can last up to five days.": "TRUE",
    "Michael Phelps has won more Olympic medals than any other athlete.": "TRUE",
    "Association football matches consist of two 45-minute halves.": "TRUE",
    "The Super Bowl is watched by more people worldwide than the FIFA World Cup final.": "FALSE",

    # ── Food & Nutrition ───────────────────────────────────────────────────
    "Carrots were originally purple before orange varieties were cultivated.": "TRUE",
    "Eating late at night directly causes weight gain regardless of total calories.": "FALSE",
    "Microwaving food destroys more nutrients than other cooking methods.": "FALSE",
    "Brown eggs are more nutritious than white eggs.": "FALSE",
    "Gluten is harmful to everyone, not just people with celiac disease.": "FALSE",
    "Alcohol directly kills brain cells.": "FALSE",
    "Spicy food causes stomach ulcers.": "FALSE",
    "Chocolate is toxic to dogs.": "TRUE",
    "Fresh vegetables are always more nutritious than frozen ones.": "FALSE",
    "Searing meat seals in its juices.": "FALSE",

    # ── Arts, Culture & Literature ─────────────────────────────────────────
    "Vincent van Gogh sold only one painting during his lifetime.": "CONTESTED",
    "William Shakespeare wrote 37 plays.": "CONTESTED",   # 36-39 by attribution
    "The Mona Lisa was stolen from the Louvre in 1911.": "TRUE",
    "Mozart died while composing his Requiem.": "TRUE",
    "The novel Frankenstein was written by Mary Shelley.": "TRUE",
    "Beethoven was completely deaf when he composed his Ninth Symphony.": "TRUE",
    "The Great Gatsby was a bestseller when first published.": "FALSE",
    "Van Gogh cut off his entire ear.": "FALSE",
    "The Sistine Chapel ceiling was painted by Michelangelo.": "TRUE",

    # ── Law & Crime ────────────────────────────────────────────────────────
    "Miranda rights originate from a 1966 US Supreme Court case.": "TRUE",
    "In the UK a person can never be tried twice for the same crime.": "FALSE",
    "The FBI was established in 1908.": "TRUE",
    "Fingerprints are unique to every individual, including identical twins.": "TRUE",
    "Interpol has the authority to arrest suspects in member countries.": "FALSE",
    "DNA evidence has exonerated hundreds of wrongfully convicted people in the United States.": "TRUE",
}

# Verdict the pipeline should return for each label.
EXPECTED_VERDICT = {"TRUE": "verified", "FALSE": "rejected"}


def score(label: str, verdict: str) -> str:
    """Classify one outcome.

    correct   — matched the reference answer
    missed    — reference is clear but the pipeline abstained (not_enough_info)
    wrong     — pipeline asserted the opposite of the reference
    contested_* — claim is genuinely disputed; abstaining is defensible
    """
    if label == "CONTESTED":
        return "contested_abstained" if verdict == "not_enough_info" else "contested_decided"
    expected = EXPECTED_VERDICT[label]
    if verdict == expected:
        return "correct"
    if verdict == "not_enough_info":
        return "missed"
    return "wrong"
