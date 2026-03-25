# Loan Eligibility Prediction


I wanted to see if I could build a model that mimics the decision-making process of a bank loan officer.
The goal is simple: Given a bunch of applicant details (like income, credit history, and education), can the machine predict if a loan should be Approved or Rejected?

---> The "Why" :

  In the real world, checking loan applications manually takes forever. I chose this project because it’s a perfect example of a Binary Classification problem. It    taught me that data isn't just numbers; it's about people’s financial lives.

---> What I Did (The Process) :

  1. Cleaning the Mess
  The raw data was pretty "leaky" (lots of missing values).
  I used the Median for numbers like LoanAmount because I didn't want a few massive loans to skew the average.
  For categories like Gender or Self_Employed, I just used the Mode (most common value).
  
  2. A Little Bit of "Brain Work" (Feature Engineering)
  One thing I realized is that banks don't just look at one person; they look at the whole household. I combined ApplicantIncome and CoapplicantIncome into a
  single Total_Income column. This actually made the model's logic a lot stronger.
  
  4. Visualizing Patterns
  I ran some EDA (Exploratory Data Analysis) and found a massive trend: Credit History is King. If an applicant had a bad credit history, the chances of approval     dropped significantly, regardless of their income.

---> The Models : 

  I didn't just pick one model; I compared two to see which was smarter:
    -Logistic Regression: My baseline. It’s simple and fast.
    -Random Forest: My "heavy lifter." I limited the depth to 2 so it wouldn't just memorize the data (overfitting) but actually learn the patterns.

---> How to Use This :

  1.)Make sure you have the loan_data.csv in the root folder.
  2.)Install the basics: pip install pandas matplotlib seaborn scikit-learn
  3.)Run the script and check the Confusion Matrix—it's the best way to see where the model gets confused!

--->Conclusion :

  This project was a great introduction to the Machine Learning. The biggest takeaway for me was that data preprocessing is just as important as the model itself.  
  By engineering the Total_Income feature and carefully handling null values, I was able to build a stable model that provides logical predictions.
