# AttendEd


A machine learning web app that predicts a student's final grade (A through F) based on academic and behavioral data, including study time, attendance, and parental support.
AttendEd (Attendance + Education) Based on the most valuble features at the data

Built as a course project by Abdelmaboud and Genidy.

## Live Demo

[attended-abdelmaboud-genidy.streamlit.app](https://attended-abdelmaboud-genidy.streamlit.app)

## Overview

The app uses a Random Forest classifier trained on a dataset of 2,392 student records. Given a student's profile, it predicts the most likely grade class and shows the probability across all five grade levels.

## Features

- Interactive sidebar to enter a student's profile (age, study time, attendance, parental support, extracurricular activities, and more)
- Instant grade prediction with confidence score
- Probability breakdown across all grade classes
- Feature importance chart showing which factors influence the model most

## Dataset

The model is trained on 13 features:

| Feature | Description |
|---|---|
| Age | Student's age |
| Gender | Male / Female |
| Ethnicity | Student's ethnicity |
| ParentalEducation | Parent's education level |
| StudyTimeWeekly | Hours spent studying per week |
| Absences | Number of absences per year |
| Tutoring | Whether the student receives tutoring |
| ParentalSupport | Level of parental support |
| Extracurricular | Participation in extracurricular activities |
| Sports | Participation in sports |
| Music | Participation in music |
| Volunteering | Participation in volunteering |
| GPA | Cumulative GPA |

## Tech Stack

- Python
- scikit-learn (Random Forest classifier)
- Streamlit (web interface)
- pandas / numpy
- Matplotlib

## Running Locally

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/attended.git
cd attended
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Project Structure

```
attended/
├── app.py             # Streamlit application
├── model.pkl           # Trained Random Forest model
├── requirements.txt    # Python dependencies
└── README.md
```

## Authors

- Abdelmaboud
- Genidy
