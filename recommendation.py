import pandas as pd

SENTIMENT_VALUES = {
    "Positive": 1.0,
    "Neutral": 0.0,
    "Negative": -1.0,
}


def get_recommendation(score):
    if score >= 0.50:
        return "Highly Recommended"
    elif score >= 0.20:
        return "Slightly Recommended"
    elif score >= -0.50:
        return "Not Recommended"
    else:
        return "Highly Not Recommended"


def get_personalized_recommendation(df, selected_areas):
    """Rank models by average review sentiment in the selected areas.

    Reviews are scored Positive=+1, Neutral=0, Negative=-1. Areas receive
    equal weight so an area with many reviews does not dominate the result.
    """
    required_columns = {"Model", "Experience_Area", "Sentiment"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Recommendation data is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    selected_areas = list(dict.fromkeys(selected_areas or []))
    if df.empty or not selected_areas:
        return None, pd.DataFrame(
            columns=["Model", "Score", "Recommendation", "Reviews"]
        )

    results = []

    for model in sorted(df["Model"].unique()):

        model_data = df[df["Model"] == model]

        area_scores = []
        review_count = 0

        for area in selected_areas:

            area_data = model_data[
                model_data["Experience_Area"] == area
            ]

            if area_data.empty:
                continue

            sentiment_values = area_data["Sentiment"].map(SENTIMENT_VALUES)
            sentiment_values = sentiment_values.dropna()
            if sentiment_values.empty:
                continue

            review_count += len(sentiment_values)
            area_score = sentiment_values.mean()

            area_scores.append(area_score)

        if not area_scores:
            continue

        # Average score across user's selected priorities
        overall_score = sum(area_scores) / len(area_scores)

        results.append({
            "Model": model,
            "Score": round(overall_score, 4),
            "Recommendation": get_recommendation(overall_score),
            "Reviews": review_count,
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        return None, results_df

    # Highest personalized score = best model
    results_df = results_df.sort_values(
        by="Score",
        ascending=False
    ).reset_index(drop=True)

    best_model = results_df.iloc[0]

    return best_model, results_df
