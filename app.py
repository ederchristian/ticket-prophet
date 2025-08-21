import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import glob
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fake Ticket Price Prophet",
    page_icon="",
    layout="wide"
)

st.title("Fake Ticket Price Prophet")
st.markdown("### FAKE DATA ENVIRONMENT - Predict if fake ticket prices will go UP or DOWN")


def get_fake_event_names():
    events = [
        'austin-fc-at-la-galaxy',
        'chargers-at-49ers',
        'dodgers-at-angels',
        'justin-bieber-in-la',
        'knicks-at-lakers',
        'olivia-rodrigo-in-sao-paulo',
        'post-malone-in-nashville',
        'sharks-at-ny-rangers',
        'taylor-swift-in-ny',
        'warriors-at-pelicans'
    ]
    return events


def get_fake_event_display_names():
    events = get_fake_event_names()
    display_names = []

    for event in events:
        display_name = event.replace('-', ' ').title()
        display_name = display_name.replace(' At ', ' at ')
        display_name = display_name.replace(' In ', ' in ')
        display_name = display_name.replace('49Ers', '49ers')
        display_name = display_name.replace('Ny ', 'NY ')
        display_name = display_name.replace('La ', 'LA ')
        display_name = display_name.replace('Fc ', 'FC ')
        display_names.append(display_name)

    return events, display_names


fake_event_names, fake_display_names = get_fake_event_display_names()


@st.cache_resource
def load_model():
    try:
        model_data = joblib.load('ticket_price_model.joblib')
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def map_fake_to_real_event(fake_event):
    fake_to_real_mapping = {
        'austin-fc-at-la-galaxy': 'seahawks-at-steelers',
        'chargers-at-49ers': 'chargers-at-49ers',
        'dodgers-at-angels': 'broncos-at-texans',
        'justin-bieber-in-la': 'shawn-mendes',
        'knicks-at-lakers': 'broncos-at-texans',
        'olivia-rodrigo-in-sao-paulo': 'lady-gaga',
        'post-malone-in-nashville': 'paul-mccartney',
        'sharks-at-ny-rangers': 'seahawks-at-steelers',
        'taylor-swift-in-ny': 'coldplay',
        'warriors-at-pelicans': 'broncos-at-texans'
    }
    return fake_to_real_mapping.get(fake_event, 'coldplay')


model_data = load_model()

if model_data is None:
    st.stop()

st.sidebar.title("Fake Data Information")
st.sidebar.markdown("""
**This is a FAKE data environment!**

This app uses the same Random Forest machine learning model as production but with fake event names for safe testing.

**Quick Mode (Recommended):**
- Just enter basic ticket info
- Model auto-generates reasonable historical data
- Fast predictions with good accuracy

**Advanced Mode:**
- Enter custom historical prices
- More precise predictions
- Better for testing specific scenarios
""")

model = model_data['model']
feature_columns = model_data['feature_columns']
le_event = model_data['le_event']
le_section = model_data['le_section']
le_row = model_data['le_row']
real_event_names = model_data['event_names']

st.header("Make Your Prediction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        event_display_selected = st.selectbox(
            "Select Event",
            options=fake_display_names,
            help="Choose from our event list"
        )

        if event_display_selected and event_display_selected in fake_display_names:
            fake_event_index = fake_display_names.index(event_display_selected)
            event_selected = fake_event_names[fake_event_index]
        else:
            event_selected = None

        available_sections = sorted(le_section.classes_)
        available_rows = sorted(le_row.classes_)

        section_selected = st.selectbox(
            "Section",
            options=available_sections,
            help="Seating section"
        )

        row_selected = st.selectbox(
            "Row",
            options=available_rows,
            help="Row within the section"
        )

        current_price = st.number_input(
            "Current Price ($)",
            min_value=1.0,
            max_value=5000.0,
            value=150.0,
            step=1.0,
            help="Current ticket price"
        )

        days_until = st.slider(
            "Days Until Event",
            min_value=1,
            max_value=365,
            value=30,
            help="How many days from now until the event"
        )

        quantity = st.number_input(
            "Tickets Available",
            min_value=1,
            max_value=1000,
            value=10,
            step=1,
            help="Number of similar tickets available"
        )

    with col2:
        advanced_mode = st.checkbox(
            "Advanced Mode",
            value=False,
            help="Enable to enter custom historical prices"
        )

        if advanced_mode:
            st.subheader("Custom Historical Data")
            st.markdown("*Optional: Enter known historical prices for better accuracy*")

            price_yesterday = st.number_input(
                "Price Yesterday ($)",
                min_value=1.0,
                max_value=5000.0,
                value=current_price * 0.98,
                step=1.0,
                help="What was the price yesterday?"
            )

            price_3_days = st.number_input(
                "Price 3 Days Ago ($)",
                min_value=1.0,
                max_value=5000.0,
                value=current_price * 1.02,
                step=1.0,
                help="What was the price 3 days ago?"
            )

            price_7_days = st.number_input(
                "Price 7 Days Ago ($)",
                min_value=1.0,
                max_value=5000.0,
                value=current_price * 0.95,
                step=1.0,
                help="What was the price 7 days ago?"
            )

            quantity_yesterday = st.number_input(
                "Quantity Yesterday",
                min_value=1,
                max_value=1000,
                value=quantity,
                step=1,
                help="How many tickets were available yesterday?"
            )

            quantity_3_days = st.number_input(
                "Quantity 3 Days Ago",
                min_value=1,
                max_value=1000,
                value=quantity * 2,
                step=1,
                help="How many tickets were available 3 days ago?"
            )

            quantity_7_days = st.number_input(
                "Quantity 7 Days Ago",
                min_value=1,
                max_value=1000,
                value=quantity * 4,
                step=1,
                help="How many tickets were available 7 days ago?"
            )
        else:
            st.subheader("Quick Prediction Mode")
            st.info("Using smart defaults for historical data. Enable Advanced Mode for manual input.")

            event_seed = hash(event_selected) % 1000000
            np.random.seed(event_seed)
            price_yesterday = current_price * np.random.uniform(0.95, 1.05)
            price_3_days = current_price * np.random.uniform(0.90, 1.10)
            price_7_days = current_price * np.random.uniform(0.85, 1.15)

            st.write(f"**Estimated Historical Prices:**")
            st.write(f"Yesterday: ${price_yesterday:.2f}")
            st.write(f"3 days ago: ${price_3_days:.2f}")
            st.write(f"7 days ago: ${price_7_days:.2f}")

            quantity_yesterday = quantity
            quantity_3_days = quantity * 2
            quantity_7_days = quantity * 4

    submitted = st.form_submit_button("Predict Price Movement", type="primary")

    if submitted:
        try:
            input_data = pd.DataFrame({
                'price': [current_price],
                'days_until_event': [days_until],
                'section': [section_selected],
                'row': [row_selected],
                'quantity': [quantity],
                'price_yesterday': [price_yesterday],
                'quantity_yesterday': [quantity_yesterday],
                'price_3_days': [price_3_days],
                'quantity_3_days': [quantity_3_days],
                'quantity_7_days': [quantity_7_days],
                'price_7_days': [price_7_days],
                'event_name_encoded': [0],
                'section_encoded': [0],
                'row_encoded': [0]
            })

            try:
                real_event = map_fake_to_real_event(event_selected)

                if real_event in le_event.classes_:
                    input_data['event_name_encoded'] = le_event.transform([real_event])
                else:
                    input_data['event_name_encoded'] = [0]

                if section_selected in le_section.classes_:
                    input_data['section_encoded'] = le_section.transform([section_selected])
                else:
                    input_data['section_encoded'] = [0]

                if row_selected in le_row.classes_:
                    input_data['row_encoded'] = le_row.transform([row_selected])
                else:
                    input_data['row_encoded'] = [0]

            except Exception as encoding_error:
                st.warning(f"Note: Using default encoding for new categories: {encoding_error}")
                input_data['event_name_encoded'] = [0]
                input_data['section_encoded'] = [0]
                input_data['row_encoded'] = [0]

            input_data = input_data[feature_columns]

            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]

            st.success("Prediction completed successfully!")

            st.header("Prediction Result")

            if prediction == 1:
                st.success("**Prediction: Price will go UP**")
                up_probability = prediction_proba[1] * 100
                down_probability = prediction_proba[0] * 100
            else:
                st.error("**Prediction: Price will go DOWN**")
                up_probability = prediction_proba[1] * 100
                down_probability = prediction_proba[0] * 100

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probability UP", f"{up_probability:.1f}%")
            with col2:
                st.metric("Probability DOWN", f"{down_probability:.1f}%")

            st.subheader("Prediction Confidence")

            prob_data = {
                'Direction': ['Price DOWN', 'Price UP'],
                'Probability': [down_probability, up_probability],
                'Color': ['red', 'green']
            }
            prob_df = pd.DataFrame(prob_data)

            fig = px.bar(prob_df, x='Direction', y='Probability', color='Color',
                        color_discrete_map={'red': '#ff4b4b', 'green': '#00ff00'},
                        title="Price Movement Probability")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Price Trend Analysis")

            price_history = {
                'Days Ago': [7, 3, 1, 0],
                'Price ($)': [price_7_days, price_3_days, price_yesterday, current_price]
            }
            price_df = pd.DataFrame(price_history)

            fig_trend = px.line(price_df, x='Days Ago', y='Price ($)',
                               title="Historical Price Trend",
                               markers=True)
            fig_trend.update_layout(xaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_trend, use_container_width=True)

            st.subheader("Input Summary")

            summary_data = {
                'Event': event_display_selected,
                'Section': section_selected,
                'Row': row_selected,
                'Current Price': f"${current_price:.2f}",
                'Days Until Event': days_until,
                'Quantity Available': quantity
            }

            for key, value in summary_data.items():
                st.write(f"**{key}:** {value}")

            st.subheader("What does this mean?")

            if up_probability > 60:
                st.info("**Strong upward trend expected!** Consider buying soon if you're planning to attend.")
            elif down_probability > 60:
                st.info("**Price likely to decrease.** You might want to wait a bit longer.")
            else:
                st.info("**Uncertain trend.** Price could go either direction with similar probability.")

        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error("Please check your input values and try again.")

st.markdown("---")
st.markdown("""
**Fake Ticket Price Prophet** - Safe testing environment with synthetic data

*Disclaimer: Predictions are estimates based on historical patterns. Actual ticket prices may vary.*
""")
