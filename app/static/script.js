document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("review-form");
  const resultDiv = document.getElementById("result");
  const sentimentP = document.getElementById("sentiment");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const reviewText = document.getElementById("review-text").value;

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: reviewText }),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      resultDiv.classList.remove("hidden");
      sentimentP.textContent = `The sentiment of this review is: ${data.sentiment}`;
    } catch (error) {
      console.error("Error:", error);
      resultDiv.classList.remove("hidden");
      sentimentP.textContent =
        "An error occurred while classifying the review.";
    }
  });
});
