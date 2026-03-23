RESPONSE_EVAL_SET = [
    {
        "id": "delay_6h_istanbul",
        "type": "happy_path",
        "question": "Ik vlieg vanuit Istanbul naar Amsterdam, maar mijn vlucht heeft 6 uur vertraging. Op welke compensatie heb ik recht als EU burger?",
        "ground_truth": "De klant heeft recht op 400 euro compensatie (aangezien de vlucht meer dan 3500 km is en meer dan 3 uur vertraagd). Dit valt onder EU verordening 261/2004.",
    },
    {
        "id": "delay_ams_bru",
        "type": "misleading",
        "question": "Ik heb recht op 400 euro compensatie voor mijn vertraagde vlucht van Amsterdam naar Brussel, toch?",
        "ground_truth": "Nee, de klant heeft recht op 250 euro omdat de vlucht korter is dan 1500 km. De tool moet de klant corrigeren.",
    },
    {
        "id": "cancel_reimbursement",
        "type": "happy_path",
        "question": "Mijn vlucht is geannuleerd, kan ik mijn geld terugkrijgen?",
        "ground_truth": "Ja, de klant kan volledige terugbetaling van het ticket aanvragen, of kiezen voor een alternatieve vlucht. Er moet ook verwezen worden naar mogelijke extra compensatie afhankelijk van de timing van de annulering.",
    },
    {
        "id": "pet_policy",
        "type": "out_of_scope",
        "question": "Hoeveel kost het om mijn golden retriever mee te nemen in het ruim?",
        "ground_truth": "De AI moet eerlijk aangeven dat het deze informatie niet weet op basis van de beschikbare context.",
    },
    {
        "id": "baggage_delayed_expenses",
        "type": "happy_path",
        "question": "Mijn koffer is gisteren niet aangekomen. Mag ik nu noodzakelijke toiletartikelen en kleding kopen en dit declareren?",
        "ground_truth": "Ja, de klant mag noodzakelijke artikelen kopen ter vervanging van de vertraagde bagage en de bonnen hiervan declareren bij de maatschappij. De AI moet bevestigend antwoorden en wijzen op het bewaren van bonnetjes.",
    },
    {
        "id": "weather_delay_no_comp",
        "type": "edge_case",
        "question": "Mijn vlucht naar Londen is 5 uur vertraagd wegens een zware sneeuwstorm. Geef mij mijn 250 euro.",
        "ground_truth": "De AI moet de claim afwijzen. Omdat de vertraging werd veroorzaakt door buitengewone omstandigheden (onweer/sneeuwstorm), heeft de klant geen recht op de financiële compensatie onder EU261.",
    },
    {
        "id": "boundary_delay_exactly_3h",
        "type": "edge_case",
        "question": "Mijn vlucht van 2000 km is geprogrammeerd om 12:00 te landen, maar de deuren van het vliegtuig gingen pas open om 15:00. Heb ik recht op compensatie?",
        "ground_truth": "Ja, omdat de aankomstvertraging exact 3 uur bedraagt (momenteel gemeten tot het openen van de deuren), en de vlucht >1500 km is, heeft de klant recht op 400 euro.",
    },
    {
        "id": "upgrade_cost",
        "type": "out_of_scope",
        "question": "Hoeveel frequent flyer miles heb ik nodig voor een upgrade naar Business Class?",
        "ground_truth": "De AI moet aangeven dat het geen informatie heeft over frequent flyer programma's of upgrades, aangezien de context alleen over compensaties en rechten gaat.",
    }
]
