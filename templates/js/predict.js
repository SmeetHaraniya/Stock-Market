document.getElementById("prediction-form").addEventListener("submit", async function(event) {
    event.preventDefault(); // Prevent form from reloading page

    const data = {
        Close: parseFloat(document.getElementById("close").value),
        High: parseFloat(document.getElementById("high").value),
        Low: parseFloat(document.getElementById("low").value),
        Open: parseFloat(document.getElementById("open").value),
        Volume: parseInt(document.getElementById("volume").value),
        Tweet_Count: parseInt(document.getElementById("tweet_count").value),
        Sentiment_Score: parseFloat(document.getElementById("sentiment").value)
    };

    const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    });

    const result = await response.json();
    document.getElementById("prediction-result").innerText = result.prediction;
});
