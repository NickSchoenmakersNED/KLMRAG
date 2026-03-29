RESPONSE_EVAL_SET = [

    # =========================================================================
    # HAPPY PATH
    # =========================================================================

    {
        "id": "delay_6h_istanbul",
        "type": "happy_path",
        "question": "Ik vlieg vanuit Istanbul naar Amsterdam, maar mijn vlucht heeft 6 uur vertraging. Op welke compensatie heb ik recht als EU burger?",
        "ground_truth": (
            "De klant heeft recht op 400 euro compensatie (contant). Istanbul-Amsterdam is circa "
            "2200 km en valt daarmee in categorie B (niet-EU vertrekpunt, tussen 1500 en 3500 km). "
            "De vertraging is meer dan 3 uur. Dit valt onder EU verordening 261/2004."
        ),
    },
    {
        "id": "cancel_reimbursement",
        "type": "happy_path",
        "question": "Mijn vlucht is geannuleerd, kan ik mijn geld terugkrijgen?",
        "ground_truth": (
            "Ja, bij annulering kan de klant kiezen tussen volledige terugbetaling van het ticket "
            "(voor het ongebruikte deel) of een alternatieve vlucht. Daarnaast heeft de klant "
            "mogelijk recht op financiële compensatie, afhankelijk van hoe laat de annulering werd "
            "meegedeeld en of er buitengewone omstandigheden waren."
        ),
    },
    {
        "id": "baggage_delayed_expenses",
        "type": "happy_path",
        "question": "Mijn koffer is gisteren niet aangekomen. Mag ik nu noodzakelijke toiletartikelen en kleding kopen en dit declareren?",
        "ground_truth": (
            "Ja, bij vertraagde ruimbagage mag de klant noodzakelijke artikelen kopen zoals kleding "
            "en toiletartikelen. De klant moet de kassabonnen bewaren en binnen 21 dagen na "
            "aflevering van de bagage een melding doen bij de maatschappij."
        ),
    },
    {
        "id": "cancel_comp_short_flight",
        "type": "happy_path",
        "question": "Mijn vlucht van Amsterdam naar Barcelona (circa 1200 km) is geannuleerd, 3 dagen voor vertrek. Ik heb geen alternatieve vlucht aangeboden gekregen. Waar heb ik recht op?",
        "ground_truth": (
            "De klant heeft recht op terugbetaling of een alternatieve vlucht, plus compensatie "
            "van 250 euro (contant) of 350 euro (voucher). De vlucht is <=1500 km (categorie A). "
            "Aangezien de annulering minder dan 7 dagen voor vertrek is meegedeeld en er geen "
            "rerouting is aangeboden, is compensatie verschuldigd."
        ),
    },
    {
        "id": "delay_care_meals",
        "type": "happy_path",
        "question": "Mijn vlucht van Parijs naar Amsterdam (kort) heeft 3 uur vertraging bij vertrek. Heb ik recht op maaltijden?",
        "ground_truth": (
            "Ja, voor vluchten van 1500 km of korter krijgt de klant assistentie (maaltijden, "
            "versnaperingen, telefoon/e-mail) bij een vertrekvertraging van 2 uur of meer. "
            "Bij 3 uur vertraging valt dit ruim binnen de drempel."
        ),
    },
    {
        "id": "cancel_14_days_notice",
        "type": "happy_path",
        "question": "KLM heeft mijn vlucht geannuleerd maar ze lieten het mij 3 weken van tevoren weten. Krijg ik compensatie?",
        "ground_truth": (
            "Nee, als de annulering minimaal 14 dagen voor de geplande vertrekdatum wordt "
            "meegedeeld, heeft de klant geen recht op financiële compensatie. Wel heeft de klant "
            "recht op terugbetaling of een alternatieve vlucht."
        ),
    },
    {
        "id": "denied_boarding_comp",
        "type": "happy_path",
        "question": "Ik ben tegen mijn wil geweigerd bij het instappen voor een vlucht van 2500 km. Wat krijg ik?",
        "ground_truth": (
            "Bij onvrijwillige instapweigering heeft de klant recht op directe compensatie op de "
            "luchthaven: 400 euro (contant) of 500 euro (voucher) voor categorie B (1500-3500 km). "
            "Daarnaast keuze tussen rerouting of terugbetaling, plus maaltijden, hotel indien "
            "nodig, en twee telefoongesprekken/e-mails."
        ),
    },
    {
        "id": "delay_5h_abandon",
        "type": "happy_path",
        "question": "Mijn vlucht heeft al 5 uur vertraging en ik wil niet meer reizen. Kan ik mijn geld terug?",
        "ground_truth": (
            "Ja, bij een vertraging van minimaal 5 uur mag de klant afzien van de reis en heeft "
            "recht op volledige terugbetaling van het ticket voor het ongebruikte deel, inclusief "
            "eventueel een retourvlucht naar het eerste vertrekpunt."
        ),
    },
    {
        "id": "downgrade_short_flight",
        "type": "happy_path",
        "question": "Ik had Business Class geboekt maar moest in Economy zitten op mijn vlucht van 1200 km. Waar heb ik recht op?",
        "ground_truth": (
            "Bij onvrijwillige downgrade op een vlucht van 1500 km of minder heeft de klant recht "
            "op terugbetaling van 30% van de ticketprijs."
        ),
    },
    {
        "id": "downgrade_long_flight",
        "type": "happy_path",
        "question": "Ik vloog van Amsterdam naar New York (5800 km) maar werd van Business naar Economy gezet. Hoeveel krijg ik terug?",
        "ground_truth": (
            "Voor vluchten buiten de EU van meer dan 3500 km heeft de klant bij onvrijwillige "
            "downgrade recht op 75% van de ticketprijs terug."
        ),
    },
    {
        "id": "baggage_damaged_airport",
        "type": "happy_path",
        "question": "Mijn koffer is zwaar beschadigd uit het vliegtuig gekomen. Wat moet ik doen?",
        "ground_truth": (
            "De klant moet zo snel mogelijk de schade melden bij de bagagebalie op de luchthaven. "
            "Er wordt dan een rapport opgesteld waarmee een vergoeding aangevraagd kan worden. "
            "Als de schade pas na vertrek van de luchthaven ontdekt wordt, kan dit nog tot 7 dagen "
            "na ontvangst van de bagage gemeld worden."
        ),
    },
    {
        "id": "baggage_lost_21_days",
        "type": "happy_path",
        "question": "Het is nu 25 dagen geleden en mijn bagage is nog steeds niet gevonden. Wat kan ik doen?",
        "ground_truth": (
            "Als de koffer niet binnen 21 dagen na aankomst van de vlucht is teruggevonden, moet "
            "de klant contact opnemen met de maatschappij. Mogelijk wordt de klant geadviseerd om "
            "een claim in te dienen bij de reisverzekering, die doorgaans het verlies dekt op "
            "basis van de waarde van de bezittingen."
        ),
    },
    {
        "id": "cancel_refund_process",
        "type": "happy_path",
        "question": "Hoe lang duurt het voordat ik mijn geld terug heb na een annulering?",
        "ground_truth": (
            "Het verwerken van een terugbetaling duurt ongeveer 2 tot 4 weken na het indienen "
            "van het verzoek via het online terugbetalingssysteem."
        ),
    },

    # =========================================================================
    # MISLEADING 
    # =========================================================================

    {
        "id": "delay_ams_bru",
        "type": "misleading",
        "question": "Ik heb recht op 400 euro compensatie voor mijn vertraagde vlucht van Amsterdam naar Brussel, toch?",
        "ground_truth": (
            "Nee, Amsterdam-Brussel is circa 170 km en valt in categorie A (<=1500 km). "
            "De contante compensatie is 250 euro, niet 400. De AI moet de klant corrigeren."
        ),
    },
    {
        "id": "misleading_weather_comp",
        "type": "misleading",
        "question": "Mijn vlucht is 4 uur vertraagd door een orkaan. Ik wil mijn 600 euro compensatie nu.",
        "ground_truth": (
            "De AI moet uitleggen dat bij buitengewone omstandigheden zoals extreem weer "
            "(orkaan) geen recht op financiële compensatie bestaat onder EU261. De klant heeft "
            "wel recht op assistentie (maaltijden, hotel, communicatie)."
        ),
    },
    {
        "id": "misleading_any_delay_comp",
        "type": "misleading",
        "question": "Mijn vlucht had 2 uur vertraging bij aankomst. Ik eis nu mijn compensatie.",
        "ground_truth": (
            "De AI moet uitleggen dat financiële compensatie pas verschuldigd is bij een "
            "aankomstvertraging van 3 uur of meer. Bij 2 uur vertraging is er geen recht op "
            "compensatie, maar mogelijk wel op assistentie afhankelijk van de vluchtafstand."
        ),
    },
    {
        "id": "misleading_600_intra_eu",
        "type": "misleading",
        "question": "Mijn vlucht van Amsterdam naar Athene (2100 km, intra-EU) is geannuleerd. Ik heb recht op 600 euro toch?",
        "ground_truth": (
            "Nee, 600 euro (contant) geldt alleen voor vluchten buiten de EU van meer dan 3500 km "
            "(categorie C). Amsterdam-Athene is een intra-EU vlucht van meer dan 1500 km, dus "
            "categorie B: 400 euro contant of 500 euro voucher."
        ),
    },

    # =========================================================================
    # EDGE CASE
    # =========================================================================

    {
        "id": "boundary_delay_exactly_3h",
        "type": "edge_case",
        "question": "Mijn vlucht van 2000 km is geprogrammeerd om 12:00 te landen, maar de deuren van het vliegtuig gingen pas open om 15:00. Heb ik recht op compensatie?",
        "ground_truth": (
            "Ja, de aankomstvertraging bedraagt exact 3 uur. Bij 3 uur of meer vertraging op "
            "aankomst heeft de klant recht op compensatie. Voor een vlucht van 2000 km "
            "(categorie B, >1500 km) is dat 400 euro contant."
        ),
    },
    {
        "id": "edge_50pct_reduction_cat_b",
        "type": "edge_case",
        "question": "Mijn vlucht van Amsterdam naar Istanbul (2200 km) is geannuleerd. Ik kreeg een alternatieve vlucht die 2,5 uur na de originele aankomsttijd aankwam. Hoeveel compensatie krijg ik?",
        "ground_truth": (
            "De vlucht is categorie B. De 50% reductie geldt als de alternatieve vlucht binnen "
            "3 uur na de originele aankomsttijd aankomt. 2,5 uur valt daarbinnen, dus de "
            "compensatie wordt gehalveerd: 200 euro contant (50% van 400) of 250 euro voucher "
            "(50% van 500)."
        ),
    },
    {
        "id": "edge_cancel_6_days_good_reroute",
        "type": "edge_case",
        "question": "Mijn vlucht is 5 dagen voor vertrek geannuleerd. KLM biedt me een vlucht aan die 45 minuten eerder vertrekt en 1,5 uur na de oorspronkelijke aankomsttijd aankomt. Compensatie?",
        "ground_truth": (
            "Nee, bij mededeling minder dan 7 dagen voor vertrek vervalt compensatie als de "
            "aangeboden vlucht maximaal 1 uur voor de geplande vertrektijd vertrekt en maximaal "
            "2 uur na de oorspronkelijke aankomsttijd aankomt. 45 minuten eerder en 1,5 uur later "
            "valt binnen die grenzen."
        ),
    },
    {
        "id": "edge_cancel_6_days_bad_reroute",
        "type": "edge_case",
        "question": "Mijn vlucht is 4 dagen voor vertrek geannuleerd. De alternatieve vlucht komt 3 uur na de originele aankomsttijd aan. Heb ik recht op compensatie?",
        "ground_truth": (
            "Ja, bij mededeling minder dan 7 dagen voor vertrek is compensatie verschuldigd als de "
            "alternatieve vlucht meer dan 2 uur na de oorspronkelijke aankomsttijd aankomt. "
            "3 uur overschrijdt die grens, dus de klant heeft recht op compensatie."
        ),
    },
    {
        "id": "edge_non_eu_scope",
        "type": "edge_case",
        "question": "Ik vloog met Emirates van Dubai naar Amsterdam en mijn vlucht had 5 uur vertraging. Heb ik recht op EU261 compensatie?",
        "ground_truth": (
            "Nee, EU261 geldt voor vluchten vanuit een niet-EU luchthaven naar de EU alleen als "
            "de uitvoerende maatschappij een EU-luchtvaartmaatschappij is (Community carrier). "
            "Emirates is geen EU-carrier, dus de verordening is niet van toepassing."
        ),
    },
    {
        "id": "edge_non_eu_klm",
        "type": "edge_case",
        "question": "Ik vloog met KLM van Dubai naar Amsterdam en mijn vlucht had 4 uur vertraging. Geldt EU261?",
        "ground_truth": (
            "Ja, KLM is een EU-carrier (Community carrier). Bij een vlucht van buiten de EU naar "
            "een EU-luchthaven met een EU-carrier is EU261 van toepassing. De klant heeft bij "
            "4 uur vertraging op aankomst recht op compensatie."
        ),
    },
    {
        "id": "edge_delay_exactly_5h_refund",
        "type": "edge_case",
        "question": "Mijn vlucht heeft precies 5 uur vertraging. Ik wil zowel mijn ticket terugbetaald als compensatie. Kan dat?",
        "ground_truth": (
            "Bij 5 uur vertraging mag de klant afzien van reizen en terugbetaling ontvangen. "
            "Daarnaast heeft de klant bij 3+ uur aankomstvertraging recht op compensatie. "
            "Beide rechten bestaan naast elkaar, tenzij buitengewone omstandigheden van "
            "toepassing zijn op de compensatie."
        ),
    },
    {
        "id": "edge_connecting_flight_missed",
        "type": "edge_case",
        "question": "Door vertraging op mijn eerste vlucht heb ik mijn aansluitende vlucht gemist. Ik wil niet meer reizen. Wat zijn mijn opties?",
        "ground_truth": (
            "Als de aansluitende vlucht niet meer haalbaar is door vertraging, kan de klant "
            "terugbetaling aanvragen voor het ongebruikte deel. Als verder reizen geen zin meer "
            "heeft, kan ook terugbetaling voor het reeds gemaakte deel plus een retourvlucht naar "
            "het eerste vertrekpunt aangevraagd worden."
        ),
    },

    # =========================================================================
    # OUT OF SCOPE 
    # =========================================================================

    {
        "id": "pet_policy",
        "type": "out_of_scope",
        "question": "Hoeveel kost het om mijn golden retriever mee te nemen in het ruim?",
        "ground_truth": (
            "De AI moet eerlijk aangeven dat het deze informatie niet weet op basis van de "
            "beschikbare context. De documenten gaan over compensatie en rechten, niet over "
            "huisdierenbeleid."
        ),
    },
    {
        "id": "upgrade_cost",
        "type": "out_of_scope",
        "question": "Hoeveel frequent flyer miles heb ik nodig voor een upgrade naar Business Class?",
        "ground_truth": (
            "De AI moet aangeven dat het geen informatie heeft over frequent flyer programma's of "
            "upgrades. De context gaat alleen over compensatie, terugbetaling en passagiersrechten."
        ),
    },
    {
        "id": "oos_booking_new_flight",
        "type": "out_of_scope",
        "question": "Ik wil een retour Amsterdam-Tokyo boeken voor volgende maand. Wat kost dat?",
        "ground_truth": (
            "De AI moet aangeven dat het geen informatie heeft over vliegticketprijzen of het "
            "boekingsproces. De beschikbare context gaat over compensatie en passagiersrechten."
        ),
    },
    {
        "id": "oos_inflight_menu",
        "type": "out_of_scope",
        "question": "Wat voor maaltijden serveren jullie aan boord op vluchten naar New York?",
        "ground_truth": (
            "De AI moet aangeven dat het geen informatie heeft over het menu aan boord. De context "
            "bevat wel informatie over terugbetaling van a-la-carte-maaltijden bij vluchtwijziging, "
            "maar niet over het reguliere aanbod."
        ),
    },
    {
        "id": "oos_seat_selection",
        "type": "out_of_scope",
        "question": "Hoe selecteer ik een stoel bij het inchecken?",
        "ground_truth": (
            "De AI moet aangeven dat het geen informatie heeft over het inchecken of "
            "stoelselectieproces. De context bevat alleen informatie over terugbetaling van "
            "stoelreserveringen, niet over het selectieproces zelf."
        ),
    },
    {
        "id": "oos_luggage_dimensions",
        "type": "out_of_scope",
        "question": "Wat zijn de maximale afmetingen voor handbagage bij KLM?",
        "ground_truth": (
            "De AI moet aangeven dat het geen informatie heeft over handbagage-afmetingen. "
            "De context gaat over compensatie bij verloren, vertraagde of beschadigde bagage, "
            "niet over bagageregels."
        ),
    },
]