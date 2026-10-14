# Local classmate recommendation pilot

The homepage now includes **Help evaluate Filmory** for signed-in accounts.
Participation is optional. The review records judged interest in recommendations,
not actual movie enjoyment or streaming completion. Play remains simulated.

## Run the pilot

1. Each classmate uses their own local account. Do not use the mapped demo account
   as evidence about a new user's preferences.
2. Complete onboarding with honest favourite genres and movies. Use normal Play,
   Like or Not interested controls as desired. Record whether the account has
   little or substantial activity when interpreting its recommendations.
3. On Home, click **Take part and review recommendations**. This explicitly opts
   into saving the exact ten-title list, ranking/model policy snapshot and ratings.
4. Judge each movie using **Looks interesting**, **Not for me** or **Not sure**.
   Unknown titles can be marked Not sure. The choices save immediately and can be
   changed before finishing. These study answers do not alter likes or history.
5. Click **Finish review**. For a comparable first pilot, ask each classmate to
   complete one review under the same instructions. Do not coach their ratings.

For local-only testing, classmates can take turns on your computer. This guide
does not require cloud hosting, paid APIs or external survey services. No
invitations or messages have been sent on your behalf.

## Export real observations

After classmates participate, choose a new output filename:

```powershell
cd E:\Filmory\backend
.\.venv\Scripts\python.exe scripts\export_recommendation_study.py --output evaluation\classmate-pilot-01.json
```

The export omits names, emails and raw account IDs, replacing IDs with stable
pseudonyms. Ratings are still stored against accounts inside the local database;
this is pseudonymization, not a claim of anonymous collection. Dedicated frontend
QA accounts are excluded automatically. No-answer or Not sure observations are
not automatically counted as negative.

The exporter reports participation, complete reviews and the share of decided
ratings marked interesting. These are descriptive pilot results. Multiple slates
from one participant are not independent users, and a small convenience sample
cannot establish general recommendation accuracy. Keep the frozen movie order
and policy snapshots with the results for reproducibility.

## Paper reporting

Report the real number of participants and completed reviews, the instructions,
activity level, abstention count, and how ratings were summarized. Use wording
such as “participants judged X% of decided suggestions interesting.” Do not label
it classifier accuracy, actual watch satisfaction, or a controlled A/B gain.

The current software QA is test evidence only. Real classmate ratings are pending
until humans submit them. Do not invent participants, percentages or comments.
