# Supply Chain Disruption Predictor Explained for Everyone

## What problem does this project solve?

This project helps a company answer a hard question before it becomes an expensive problem:

"Which supplier orders are likely to arrive late, and what should we do about them right now?"

In many companies, especially manufacturers, late deliveries create a chain reaction:

- Raw materials arrive late.
- Production slows down or stops.
- Teams rush to find alternatives.
- Emergency buying becomes expensive.
- Customers receive orders late.
- Finance absorbs avoidable cost.

Most teams discover these problems after the delay has already happened. That is reactive management. This project is built to support proactive management.

Instead of waiting for a supplier to fail, the system looks at the warning signs early and estimates which open orders carry the highest risk.

## Why is this useful in plain language?

Think of this system like a weather forecast for supplier deliveries.

It does not magically stop the rain. What it does is tell you when there is a strong chance of rain so you can carry an umbrella, change your route, or delay travel.

This project works the same way:

- It does not guarantee that an order will be late.
- It estimates the probability of lateness.
- It turns that probability into a practical risk label.
- It helps a procurement team decide where to focus first.

That matters because procurement teams usually have limited time, limited attention, and many orders running at once. They cannot chase every supplier every day. They need a ranked list of where intervention is most valuable.

## What business questions does it answer?

This project is designed to answer questions like these:

1. Which pending purchase orders look risky today?
2. Which suppliers are becoming more unreliable over time?
3. Are external conditions making delays more likely?
4. Is a supplier's internal condition adding risk?
5. Which orders deserve escalation first?
6. Where might the business face avoidable cost if it does nothing?

## What decisions can be made from this?

The project does not make decisions automatically. It supports human decisions.

Here are the kinds of actions a procurement, operations, or supply chain team could take from its output.

### Order-level decisions

- Escalate a high-risk order to supplier management.
- Call the supplier earlier for a delivery status check.
- Ask for partial shipment instead of full shipment.
- Expedite transport for a critical order.
- Delay dependent production activity until risk is clearer.
- Re-prioritize internal planning around at-risk materials.

### Supplier-level decisions

- Review suppliers with repeated late-risk patterns.
- Negotiate stronger service terms.
- Shift some volume away from high-risk suppliers.
- Add backup suppliers for fragile categories.
- Increase monitoring frequency for risky suppliers.

### Financial and planning decisions

- Estimate which risky orders expose the most money.
- Focus mitigation on high-value orders first.
- Defend requests for contingency budget.
- Show management where disruption cost may occur.
- Compare the cost of acting now versus waiting.

## What data does the project use?

The project combines three kinds of information.

### 1. Purchase order data

This is the core transaction data. It tells the system things like:

- who the supplier is
- when the order was placed
- expected delivery date
- actual delivery date for completed orders
- order value
- order quantity
- whether the order is delivered, partial, or still pending

This is the historical record the model learns from.

### 2. Supplier health data

This represents the condition of the supplier itself. In simple terms, it asks:

"How stable and operationally healthy does this supplier look?"

The project simulates indicators such as:

- financial health score
- production capacity utilization
- employee turnover
- quality defect rate
- inventory days
- payment default risk

These signals matter because suppliers do not fail randomly. Weak finances, high turnover, poor quality, and strained capacity often show up before delivery failure.

### 3. External risk data

This represents the outside world around the supplier and the transport network.

Examples include:

- rainy season
- rainfall level
- holiday period
- fuel price pressure
- exchange rate pressure
- port congestion
- disruption events such as strikes or weather events

These signals matter because even a good supplier can deliver late when the surrounding environment becomes unstable.

## How does the project solve the problem?

The system solves the problem in stages. Each stage adds context until the final decision support output is ready.

### Step 1: Read the core business records

The pipeline starts by loading supplier and purchase order data.

Before doing anything else, it checks that required columns are present. This is important because if key fields are missing, every later result becomes unreliable.

In simple terms, this is the project asking:

"Do I have the basic facts needed to reason about supplier risk?"

### Step 2: Build the outside-world context

The project generates daily external risk conditions across the same date range as the purchase orders.

This is a crucial detail.

If your purchase orders are from one time period, but your weather and disruption data are generated for a different time period, the join fails and the model learns from blanks. That exact issue existed earlier in the project and was fixed by aligning dates correctly.

That fix matters because it restored the project from pretending to use external conditions to actually using them.

### Step 3: Build the supplier condition context

The project creates monthly supplier health records for each supplier.

This gives each order more context. Instead of seeing only an order value and date, the system also sees a rough picture of the supplier's operating condition during that month.

This is similar to evaluating a football team. You would not only look at the next match. You would also look at the team's fitness, morale, injuries, and recent form.

### Step 4: Join everything to each order

The system merges the purchase order record with:

- the external risk signals for that order date
- the supplier health signals for that supplier and month

After joining, it checks coverage. That means it measures how much of the expected information actually attached to each order.

This step exists because missing context can silently damage a model. The project explicitly checks for that instead of assuming the merge worked.

### Step 5: Measure whether past orders were late

For completed deliveries, the project calculates delivery delay days by comparing actual delivery date with expected delivery date.

If the delivery happened after the promised date, the order is labeled late.

This late flag becomes the target the model tries to learn.

In plain language, the model studies past orders and learns the patterns that tended to happen before lateness.

### Step 6: Create a simple risk score people can understand

The project does not only rely on machine learning. It also creates a composite risk score from 0 to 100.

That score blends three sources of risk:

- supplier-based risk
- external risk
- operational risk

The default weighting is:

- 40% supplier
- 30% external
- 30% operational

Why do this if there is already a machine learning model?

Because a composite score is easier for people to explain in meetings. It acts like a transparent summary signal. The machine learning model adds predictive power, but the score gives decision-makers something more interpretable.

### Step 7: Create memory-based features

The project then creates features that reflect recent behavior. These include:

- average delay over recent orders
- late delivery rate over recent orders
- days since the last order
- current order size compared with that supplier's average order size

This is the system asking:

"What has this supplier been doing lately, and is this new order unusual?"

That matters because recent behavior often predicts near-future behavior better than older history does.

### Step 8: Train a model on completed orders

The model used here is a Random Forest classifier.

For a layman, you can think of it as a large panel of decision trees voting together. Each tree looks at the data a little differently. The final answer combines many trees instead of trusting one brittle rule.

The model learns from completed orders where the outcome is already known.

It uses 18 features that include supplier condition, external conditions, order size, order history, and the composite risk score.

### Step 9: Test whether the model is learning anything useful

The project does not stop at training. It measures performance using:

- accuracy
- precision
- recall
- F1 score
- ROC-AUC
- cross-validation

Why so many metrics?

Because one metric can be misleading.

For example, if late deliveries are common, a model can look acceptable on accuracy while still doing a poor job of distinguishing risky orders. That is why the project keeps several views of model quality.

### Step 10: Score open orders

Once the model is trained, it is used on pending orders where the final outcome is not known yet.

For each pending order, the system estimates the probability of lateness.

It then translates that number into a simple label:

- Low Risk
- Medium Risk
- High Risk

This turns a technical output into an operational queue.

### Step 11: Create supplier scorecards

The project also builds a supplier-level scorecard.

This scorecard combines:

- historical late rate
- average delay days
- predicted lateness on pending orders
- current supplier health
- quality defect rate

This helps the business move from reacting to one bad order toward managing supplier relationships more strategically.

### Step 12: Estimate business impact

The project calculates a scenario-based estimate of possible disruption cost and possible savings from intervention.

This is not presented as a guaranteed outcome. It is clearly labeled as assumption-based.

That distinction matters.

The system is saying:

"If these risky orders are delayed, and if delay usually costs this much, and if intervention prevents some of that loss, then this is the rough financial exposure."

That makes the output useful for finance and operations planning, while remaining honest about what is estimated versus measured.

### Step 13: Validate claims made in documentation

The project includes a claim validator that checks whether headline statements in the documentation match the actual artefacts.

This is unusual and valuable.

It means the project does not just produce charts and numbers. It also checks whether the story being told about the project is true.

## What is the core logic behind how everything works?

The logic is simple at a high level:

1. Past delivery outcomes contain patterns.
2. Those patterns are influenced by supplier condition, operational strain, and external disruptions.
3. If we measure those signals consistently, we can estimate the chance of a future delay.
4. If we estimate that chance early enough, teams can intervene.
5. If teams intervene on the right orders, they can reduce avoidable cost and disruption.

That is the full logic chain.

The project is not trying to predict the future with certainty. It is trying to improve decision quality under uncertainty.

## An example in everyday terms

Imagine two pending orders:

### Order A

- comes from a supplier with decent financial health
- has low recent late-delivery history
- is placed in a relatively calm external environment
- has normal order size

The system may classify this as Low Risk.

### Order B

- comes from a supplier with weak health metrics
- has had several recent late deliveries
- is happening during heavy congestion and disruption conditions
- is a large order that is important to production

The system may classify this as High Risk.

What do those labels mean in practice?

They help the team decide that Order B deserves immediate attention, while Order A may only need normal monitoring.

## What makes this project credible?

Several design decisions improve trust:

- It checks data contracts before processing.
- It checks join coverage after merging data.
- It separates technical metrics from financial assumptions.
- It records reproducibility metadata.
- It uses test cases and CI.
- It validates public claims against generated artefacts.

These may sound like engineering details, but they are really trust-building steps.

## What are the current results?

At the time of writing, the saved artefacts show:

- test accuracy of about 54.5%
- recall for late deliveries of about 63.2%
- ROC-AUC of about 0.540
- 2,326 completed training rows
- 6 currently high-risk pending orders in the scenario output
- about NGN 36.8M in scenario-based potential savings under stated assumptions

These results should be interpreted carefully.

They show that the system can learn some signal, but it is not yet a production-grade oracle. It is a strong baseline and a useful decision-support prototype.

## What are the limitations?

To explain this honestly to a non-technical audience, here is the simple truth:

- The project uses synthetic data, not live production data.
- The business impact numbers are estimates, not measured savings.
- The model has moderate predictive strength, not high predictive strength.
- The train/test split is random, not a full time-based forecasting setup.
- The system supports decisions; it should not replace human judgment.

## What should a business leader take away from this?

If you are a business leader, the main takeaway is this:

You do not need perfect foresight to make better operational decisions. You need a ranked view of where risk is building so teams can act earlier.

This project demonstrates that idea in a structured way.

It turns scattered signals into:

- order-level warnings
- supplier-level risk views
- measurable model outputs
- scenario-based cost discussion

That makes it easier for a business to move from intuition-only procurement toward evidence-guided procurement.

## What should a recruiter, hiring manager, or stakeholder take away?

This project shows the ability to:

- frame a real business problem clearly
- convert business signals into usable features
- build a full data and ML pipeline
- think about explainability, testing, and documentation
- distinguish between measured facts and estimated claims
- design outputs that support practical business action

## One-sentence summary

This project is an early warning system that helps a company spot which supplier orders are most likely to arrive late, understand why, and decide where to intervene first.