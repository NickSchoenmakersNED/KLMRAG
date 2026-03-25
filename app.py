import streamlit as st
import main

st.set_page_config(page_title="KLM Passenger Rights Assistant", page_icon="✈️")

st.title("KLM Passenger Rights Assistant ✈️")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "current_classification" not in st.session_state:
    st.session_state.current_classification = main.QueryClassification()
    
if "accumulated_question" not in st.session_state:
    st.session_state.accumulated_question = ""
    
if "asking_followup" not in st.session_state:
    st.session_state.asking_followup = False

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        clf = st.session_state.current_classification
        
        if st.session_state.asking_followup:
            st.session_state.accumulated_question = f"{st.session_state.accumulated_question}\n{prompt}"
            llm_clf = main.classify_query(prompt, current=clf)
            clf = main.merge_classification(clf, llm_clf)
            st.session_state.current_classification = clf
        else:
            st.session_state.accumulated_question = prompt
            clf = main.classify_query(prompt)
            st.session_state.current_classification = clf
            
        missing = main.get_missing_fields(clf)
        
        if missing:
            st.session_state.asking_followup = True
            response = main.ask_followup(st.session_state.accumulated_question, missing)
        else:
            st.session_state.asking_followup = False
            
            flight_info_note = ""
            if clf.has_flight_number and clf.flight_number:
                info = main.fetch_flight_info(clf.flight_number)
                if not clf.has_origin:
                    clf.has_origin = True
                    clf.origin = info["origin"]
                if not clf.has_destination:
                    clf.has_destination = True
                    clf.destination = info["destination"]
                if not clf.has_hours:
                    clf.has_hours = True
                    clf.hours_value = str(info["delay_hours"])
                flight_info_note = (
                    f"\n\n[Systeem Notitie: De API heeft de volgende vluchtinformatie gevonden voor {clf.flight_number}:\n"
                    f"- Vertrek: {info['origin']}\n"
                    f"- Aankomst: {info['destination']}\n"
                    f"- Geplande vertrektijd: {info['scheduled_departure']}\n"
                    f"- Werkelijke vertrektijd: {info['actual_departure']}\n"
                    f"- Geplande aankomsttijd: {info['scheduled_arrival']}\n"
                    f"- Werkelijke aankomsttijd: {info['actual_arrival']}\n"
                    f"- Vertraging bij aankomst: {info['delay_hours']} uur\n"
                    f"- Reden vertraging: {info['reason']}]"
                )

            distance_info = ""
            if clf.has_origin and clf.has_destination:
                distance = main.calculate_distance(clf.origin, clf.destination)
                if distance:
                    distance_info = f"\n\n[Systeem Notitie: De berekende vliegafstand van {clf.origin} naar {clf.destination} is ongeveer {distance:.0f} km. Gebruik deze afstand om te bepalen in welke categorie de vlucht valt volgens de regels.]"

            rewritten = main.rewrite_query(st.session_state.accumulated_question, clf=clf)
            docs = main.retriever.invoke(rewritten)
            context = main.format_docs_with_sources(docs)
            
            augmented_question = st.session_state.accumulated_question + flight_info_note + distance_info

            raw_answer = main.generation_chain.invoke({
                "context": context,
                "question": augmented_question,
                "history": main.format_history(main.conversation_history),
            })
            response = main.finalize_response_with_sources(raw_answer, docs)
            main.conversation_history.append({
                "question": prompt,
                "answer": response,
            })
            
            # Reset state for next complete question
            st.session_state.current_classification = main.QueryClassification()
            st.session_state.accumulated_question = ""
            
        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
