"""
eval_dataset.py

Ground truth pairs: (question, list of strings that MUST appear in a retrieved chunk).
These are hand-labeled. The keyword list should be specific enough to identify the
right chunk but not so brittle that minor rewording breaks it.

Two entry types:
  - Standard entries: retrieval must return a chunk containing any expected_keywords
  - out_of_scope: True entries: retrieval must NOT return chunks containing any of
    expected_keywords (which are in-scope terms). Handled separately in the tester.

Question types:
  - happy_path    : Straightforward, well-formed, correct-premise questions
  - misleading    : Wrong premise baked into the question
  - noisy_input   : Misspelled, spoken/transcribed, or grammatically poor input
  - edge_case     : Boundary conditions, unusual-but-valid scenarios
  - out_of_scope  : Topic not covered in source documents at all

Add more entries as you find retrieval failures.
"""

EVAL_SET = [
    # ── Dutch ─────────────────────────────────────────────────────────────────

    {
        "id": "delay_6h_compensation",
        "type": "happy_path",
        "question": "Ik vlieg vanuit Istanbul naar Amsterdam, maar mijn vlucht heeft 6 uur vertraging. Op welke compensatie heb ik recht?",
        "expected_keywords": ["400", "artikel 7", "compensatie"],
        "notes": "Long-haul >3500km, delay >3h → €400 under EC 261/2004 Art.7"
    },
    {
        "id": "delay_6h_compensation_misspelled",
        "type": "noisy_input",
        "question": "k heb n vluicht vanuit istanbul na amsterdam mar mijn vluch heft 6 uut vertraaging op welke compensatir heb ik reht",
        "expected_keywords": ["400", "artikel 7", "compensatie"],
        "notes": "Intentionally misspelled query. Same premise as delay_6h_compensation."
    },
    {
        "id": "delay_6h_compensation_story",
        "type": "noisy_input",
        "question": "hey, ik was dus op vakantie aan het gaan naar istanbul. naar de hagia sofia gegaan en alles, was wel leuk. maar toen brak uiteindelijk de dag aan waarin ik m'n familie naar huis moesten vliegen, jammer. maar ja. dus, wij komen aan op het vliegveld. druk natuurlijk, en de kinderen weer in histerie voor een of andere reden. en dan komen wij uiteindelijk aan, en dan is hij vertraagd. dit vind ik natuurlijk niet echt pretig. had nog een afspraak gemaakt gisteren met een vriend die bij de KLM werkt. had hem al een bericht gestuurd over wat de regels allemaal zijn. maar dat duurt mij eigenlijk een beetje te lang. maar in ieder geval. het terminal ding zegt dat ik in totaal 6 uur vertraging ga hebben. dus eingelijk vind ik wel dat ik compensatie verdien. maar ik weet niet hoeveel.",
        "expected_keywords": ["400", "artikel 7", "compensatie"],
        "notes": "Long-winded informal version of delay_6h_compensation."
    },
    {
        "id": "delay_6h_compensation_spoken",
        "type": "noisy_input",
        "question": "ik vlieg van uit istanbul naar amsterdam maar mijn vlu ik vlieg vanuit istanbul naar amsterdam en de vlucht heeft zes uur vertraging mama even wachten even wachten op welke compensatie heb ik recht of wat voor compensatie krijg ik daar voor",
        "expected_keywords": ["400", "artikel 7", "compensatie"],
        "notes": "Emulates spoken/transcribed language. Same premise as delay_6h_compensation."
    },
    {
        "id": "delay_6h_compensation_poor_dutch",
        "type": "noisy_input",
        "question": "ik vlieg instanbul amsterdam. de vliegtuig is 6 uur langer. krijg ik geld?",
        "expected_keywords": ["400", "artikel 7", "compensatie"],
        "notes": "Intentionally poor Dutch. Same premise as delay_6h_compensation."
    },
    {
        "id": "delay_threshold",
        "type": "happy_path",
        "question": "Vanaf hoeveel uur vertraging heb ik recht op compensatie?",
        "expected_keywords": ["3 uur", "aankomsttijd", "vertraging"],
        "notes": "Threshold is arrival delay >= 3h. Tests retrieval of the Sturgeon rule."
    },
    {
        "id": "delay_under_3h",
        "type": "edge_case",
        "question": "Mijn vlucht had 2,5 uur vertraging bij aankomst. Heb ik recht op compensatie?",
        "expected_keywords": ["3 uur", "aankomsttijd"],
        "notes": "Arrival delay must be >= 3h. 2.5h does not qualify."
    },
    {
        "id": "delay_5h_refund",
        "type": "happy_path",
        "question": "Mijn vlucht heeft al 5 uur vertraging en ik wil helemaal niet meer vliegen. Kan ik mijn geld terugkrijgen?",
        "expected_keywords": ["5 uur", "terugbetaling", "ticket"],
        "notes": ">= 5h delay triggers right to full reimbursement and return flight."
    },
    {
        "id": "delay_care_obligations",
        "type": "happy_path",
        "question": "Mijn vlucht is vertraagd. Moet de luchtvaartmaatschappij mij eten en drinken geven?",
        "expected_keywords": ["maaltijden", "versnaperingen", "2 uur"],
        "notes": "Care obligations (meals, refreshments) kick in at 2h departure delay."
    },
    {
        "id": "delay_50pct_reduction",
        "type": "edge_case",
        "question": "Ik vlieg van Amsterdam naar Dubai en mijn vlucht is 3,5 uur vertraagd bij aankomst. Krijg ik de volle compensatie?",
        "expected_keywords": ["50%", "3.500 km"],
        "notes": ">3500km, arrival delay 3-4h -> 50% reduction applies. 300 not 600."
    },
    {
        "id": "delay_weather_exception",
        "type": "edge_case",
        "question": "Mijn vlucht was vertraagd door een zware storm. Heeft de luchtvaartmaatschappij dan nog steeds een betalingsplicht?",
        "expected_keywords": ["buitengewone omstandigheden", "extreem weer"],
        "notes": "Extraordinary circumstances (weather) exempt the airline from compensation."
    },
    {
        "id": "extraordinary_circumstances",
        "type": "happy_path",
        "question": "Wat zijn buitengewone omstandigheden die compensatie uitsluit?",
        "expected_keywords": ["buitengewone omstandigheden", "extreem weer", "politieke instabiliteit"],
        "notes": "General query about extraordinary circumstances exemptions."
    },
    {
        "id": "cancellation_rights",
        "type": "happy_path",
        "question": "Mijn vlucht is geannuleerd, wat zijn mijn rechten?",
        "expected_keywords": ["annulering", "terugbetaling", "alternatieve vlucht"],
        "notes": "Cancellation -> refund or rerouting + care obligations."
    },
    {
        "id": "cancellation_rerouting_options",
        "type": "happy_path",
        "question": "Mijn vlucht is geannuleerd. Kan ik kiezen tussen een andere vlucht of mijn geld terug?",
        "expected_keywords": ["alternatieve vlucht", "terugbetaling", "keuze"],
        "notes": "Core choice: rerouting vs full reimbursement."
    },
    {
        "id": "cancellation_reimbursement_timeframe",
        "type": "happy_path",
        "question": "Binnen hoeveel dagen moet ik mijn geld terugkrijgen als mijn vlucht geannuleerd is?",
        "expected_keywords": ["zeven dagen"],
        "notes": "Specific threshold: reimbursement within 7 days."
    },
    {
        "id": "cancellation_wrong_premise_no_hotel",
        "type": "misleading",
        "question": "Als mijn vlucht geannuleerd is hoef ik zelf het hotel te betalen toch? De maatschappij is daar niet verantwoordelijk voor.",
        "expected_keywords": ["hotelaccommodatie", "kosteloos", "nacht"],
        "notes": "Wrong premise. Carrier IS obligated to cover hotel if overnight stay is necessary."
    },
    {
        "id": "downgrading_short_haul",
        "type": "happy_path",
        "question": "Ik heb een businessclass ticket gekocht maar werd in economy gezet op een vlucht van Amsterdam naar Parijs. Wat krijg ik terug?",
        "expected_keywords": ["30%", "ticketprijs", "1.500 km"],
        "notes": "AMS-CDG <1500km -> 30% of ticket price refund."
    },
    {
        "id": "downgrading_long_haul",
        "type": "happy_path",
        "question": "Ik zat in een lagere klasse dan ik had geboekt op mijn vlucht van Amsterdam naar New York. Hoeveel procent van mijn ticket krijg ik terug?",
        "expected_keywords": ["75%", "ticketprijs"],
        "notes": "AMS-JFK >3500km -> 75% of ticket price."
    },
    {
        "id": "downgrading_wrong_premise",
        "type": "misleading",
        "question": "Als ik onvrijwillig in een lagere klasse word geplaatst krijg ik mijn volledige ticket terugbetaald, toch?",
        "expected_keywords": ["30%", "50%", "75%"],
        "notes": "Wrong premise. Refund is a percentage, not full reimbursement."
    },
    {
        "id": "denied_boarding_compensation",
        "type": "happy_path",
        "question": "Ik had een bevestigde boeking maar werd toch niet toegelaten tot de vlucht omdat die overvol was. Heb ik recht op compensatie?",
        "expected_keywords": ["instapweigering", "compensatie", "vrijwilligers"],
        "notes": "Involuntary denied boarding -> compensation. Tests whether overbooked context is retrieved."
    },
    {
        "id": "denied_boarding_voluntary",
        "type": "edge_case",
        "question": "De luchtvaartmaatschappij vroeg of iemand zijn stoel wilde afstaan. Ik heb ja gezegd. Krijg ik dan hetzelfde als iemand die gedwongen van boord gaat?",
        "expected_keywords": ["vrijwillig", "overeengekomen", "vrijwilligers"],
        "notes": "Voluntary surrender -> negotiated voucher, not mandatory compensation amounts."
    },
    {
        "id": "denied_boarding_no_right",
        "type": "edge_case",
        "question": "Ik werd geweigerd bij de gate omdat mijn paspoort verlopen was. Heb ik recht op compensatie voor instapweigering?",
        "expected_keywords": ["reispapieren", "onvolledige"],
        "notes": "Incomplete travel documents exempt the airline from compensation."
    },
    {
        "id": "maximum_compensation",
        "type": "happy_path",
        "question": "Wat is de maximale vergoeding die ik kan krijgen?",
        "expected_keywords": ["600", "800"],
        "notes": "Max cash compensation is 600, max voucher is 800."
    },
    {
        "id": "wrong_premise_1000_compensation",
        "type": "misleading",
        "question": "Ik heb gehoord dat ik 1000 euro krijg voor mijn vertraagde vlucht. Waar kan ik dit ophalen?",
        "expected_keywords": ["250", "400", "600"],
        "notes": "Wrong premise. 1000 does not exist as a compensation amount. Should surface actual tiers."
    },
    {
        "id": "leading_question_400_ams_bru",
        "type": "misleading",
        "question": "Ik heb recht op 400 euro compensatie voor mijn vertraagde vlucht van Amsterdam naar Brussel, toch?",
        "expected_keywords": ["1.500 km", "250"],
        "notes": "Wrong premise. AMS-BRU <1500km -> 250, not 400."
    },
    {
        "id": "wrong_airline",
        "type": "edge_case",
        "question": "Ik ben via KLM naar Polen gevlogen maar nu vlieg ik terug via Wizzair. Mijn Wizzair vlucht is vertraagd. Kan ik compensatie krijgen?",
        "expected_keywords": ["uitvoerende luchtvaartmaatschappij", "had moeten uitvoeren"],
        "notes": "The KLM outbound leg is irrelevant. Claim is against the operating carrier."
    },
    {
        "id": "third_party_delay_missed_klm",
        "type": "edge_case",
        "question": "Mijn Wizz Air-vlucht had vertraging, waardoor ik mijn aansluitende KLM-vlucht heb gemist. Heb ik recht op compensatie van KLM?",
        "expected_keywords": ["uitvoerende luchtvaartmaatschappij", "aansluitende vlucht"],
        "notes": "KLM is not liable for disruptions caused by a third-party carrier."
    },
    {
        "id": "out_of_scope_dog",
        "type": "out_of_scope",
        "question": "Ik wil mijn hond meenemen op de vlucht. Kan dat?",
        "expected_keywords": ["compensatie", "terugbetaling", "artikel 7"],
        "notes": "No pet travel information in any source document."
    },
    {
        "id": "out_of_scope_booking",
        "type": "misleading",
        "question": "Ik wil een vlucht van Eindhoven naar Canberra boeken en wil maximaal 400 euro uitgeven. Kun jij dit voor mij regelen?",
        "expected_keywords": ["compensatie", "terugbetaling", "artikel 7"],
        "notes": "Booking assistance is not covered in any source document."
    },
    {
        "id": "out_of_scope_personal_injury",
        "type": "out_of_scope",
        "question": "Door een defecte trede in de trap op een KLM-vliegtuig heb ik een gebroken been opgelopen. Hoe kan ik compensatie claimen bij KLM?",
        "expected_keywords": ["compensatie", "terugbetaling", "artikel 7"],
        "notes": "Personal injury falls under Montreal Convention. Not covered in source documents."
    },

    # ── English ────────────────────────────────────────────────────────────────

    # ── German ────────────────────────────────────────────────────────────────
]