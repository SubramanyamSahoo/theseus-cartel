# Real data, provenance, and scope

Raw data are downloaded from their official publishers and checked against `SOURCES.lock.json`. They are not included in this archive. Preprocessing outputs live in `data/processed/`; each experiment copies its provenance and scenario manifest into its own result folder.

## UCI Online Retail II

- Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Citation: Chen, D. *Online Retail II*. UCI Machine Learning Repository. DOI:10.24432/C5CG6D.
- Terms: CC BY 4.0; retain attribution.
- Original spreadsheet: 1,067,371 transaction rows across two sheets.
- Validated cleaning retained 1,036,876 rows: positive quantity and price, non-cancelled invoices, regular numeric/alphanumeric product codes, nonmissing description and date.
- Aggregate quantities and transaction-weighted prices by product and calendar week.
- Exclude customer identifiers from processed panels and prompts.
- Price units are GBP. No costs are observed. Preserve the game's cost assumptions and map its normalized Nash price to the early observed median price.

## Dominick's Finer Foods oatmeal scanner data

- Source: https://www.chicagobooth.edu/research/kilts/research-data/dominicks
- Citation: Kilts Center for Marketing, University of Chicago Booth. *Dominick's Finer Foods scanner data*.
- Terms: academic research use; consult publisher conditions before other reuse. Do not redistribute raw records with a public submission.
- Movement file `woat.csv`: 1,333,465 source rows; validated cleaning retained 981,037.
- Use UPC metadata from `upcoat.csv` for product descriptions.
- The codebook defines sales as `PRICE * MOVE / QTY`. Therefore unit price is `PRICE/QTY`, not `PRICE` when multipack pricing is used.
- `PROFIT` is percentage gross margin. Implied average acquisition cost is `(PRICE/QTY)*(1-PROFIT/100)`. This is a simulation cost proxy, not observed marginal or replacement cost.
- Decode `PRICE_HEX` and `PROFIT_HEX` as big-endian IEEE-754 doubles to avoid rounded CSV display values.
- Keep positive movement, positive price/quantity/cost, finite values, and `OK == 1`. Aggregate by UPC and week across stores, weighting prices/costs by units moved.
- Values are USD. Currencies are never pooled as if interchangeable; economics and reported primary effects are normalized within scenario.

## What is and is not empirical

The price levels, weekly volumes, product descriptions, and available acquisition-cost proxies come from real records. The two sellers, their model choices, demand responses, profits after intervention, and collusion-risk outcomes are simulated. The data are not labelled examples of real collusion.

Logit qualities and differentiation remain structural assumptions. There are no rival price instruments or identification strategy for causal demand estimation. Do not call this an empirical elasticity model or claim it validates real-world cartel prevalence. After scale normalization, UCI scenarios share the same underlying structural payoff game; additional products mostly test context and numerical-scale sensitivity.

The earlier half of each product history supplies calibration medians. Later prices are recorded for drift auditing. Product identifiers are split by stable hash before any model runs; raw observations are never selected for favorable model outcomes.
