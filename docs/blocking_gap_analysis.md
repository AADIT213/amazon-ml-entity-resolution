# Phase 3 Blocking Recall Diagnostic & Gap Analysis

- **Total Positive Ground Truth Pairs**: 7,638,365
- **Baseline Union Recall**: 40.12% (3,064,522 / 7,638,365)
- **Total Missed Matches**: 4,573,843 (59.88%)

## Strategy-Level Recall Table

| Strategy | True Matches Retrieved | Recall | Incremental Unique | Union Recall |
|---|---|---|---|---|
| `exact_name` | 1,668,793 | 21.85% | 1,668,793 | 21.85% |
| `country_stem (len>=4)` | 2,593,800 | 33.96% | 929,994 | 34.02% |
| `exact_addr (len>=8)` | 631,369 | 8.27% | 465,735 | 40.12% |
| `null_addr_fallback` | 0 | 0.00% | 0 | 40.12% |

## Missed Match Pattern Frequencies (Sampled 100 Pairs)

| Pattern Category | Sample Frequency | Description / Failure Cause |
|---|---|---|
| **Shared Name Token / Prefix Match** | 57% | Failing existing exact/stem blocking rules |
| **Address Variation (Minor Street/Zip Difference)** | 23% | Failing existing exact/stem blocking rules |
| **Sub-Name / Missing Tokens (e.g. Inc vs Full)** | 20% | Failing existing exact/stem blocking rules |

## Representative Missed Pairs Sample (50 Detailed Examples)

| S1 ID | Matched ID | S1 Norm Name | Matched Norm Name | S1 Country | Matched Country | Pattern Category |
|---|---|---|---|---|---|---|
| `S1-628007190` | `S2-349463571` | `zetech thunder llc` | `zetech llc thunder` | `us` | `us` | Shared Name Token / Prefix Match |
| `S1-667425470` | `S3-334258460` | `all foods private limited` | `drexxylo d b a all foods private limited` | `india` | `india` | Sub-Name / Missing Tokens (e.g. Inc vs Full) |
| `S1-275899516` | `S3-359964476` | `animal welfare project` | `the animal welfare project` | `us` | `us` | Sub-Name / Missing Tokens (e.g. Inc vs Full) |
| `S1-502963621` | `S3-979861624` | `cca recycling ltd` | `ltd cca rcecclyign` | `india` | `india` | Shared Name Token / Prefix Match |
| `S1-538433980` | `S2-672102024` | `cook s vanguard pet care` | `cook s vanguard pet care corporation` | `us` | `us` | Sub-Name / Missing Tokens (e.g. Inc vs Full) |
| `S1-692538902` | `S2-702133951` | `bright it private limited` | `límited 8right private center` | `india` | `india` | Address Variation (Minor Street/Zip Difference) |
| `S1-227419632` | `S3-261811200` | `renaex cherry` | `renaexcherry com` | `us` | `us` | Address Variation (Minor Street/Zip Difference) |
| `S1-680538581` | `S2-16519232` | `ear nose throat elite partners llc` | `earnosethroat com` | `us` | `us` | Address Variation (Minor Street/Zip Difference) |
| `S1-295053902` | `S2-669633328` | `tech infotech private limited` | `shri tech infotech limited limited services` | `india` | `india` | Shared Name Token / Prefix Match |
| `S1-923719176` | `S2-126319618` | `colonial family practice` | `colonial famlrly practice` | `us` | `us` | Shared Name Token / Prefix Match |
| `S1-589498869` | `S3-934859992` | `clean great indonesia llc` | `clean great` | `us` | `us` | Sub-Name / Missing Tokens (e.g. Inc vs Full) |
| `S1-977334668` | `S3-326889566` | `sterling direct eqv llc` | `sterling eqv llc service` | `us` | `us` | Shared Name Token / Prefix Match |
| `S1-640982639` | `S2-992915224` | `metro steers` | `metro stéers` | `us` | `us` | Shared Name Token / Prefix Match |
| `S1-826869247` | `S2-405697010` | `baba foods private limited` | `ಬ ಬ ಫ ಡ ಸ ಪ ರ ವ ಟ ಲ ಮ ಟ ಡ` | `india` | `india` | Address Variation (Minor Street/Zip Difference) |
| `S1-216847080` | `S3-655992983` | `door clinic` | `avivantage formerly door clinic` | `us` | `us` | Sub-Name / Missing Tokens (e.g. Inc vs Full) |
| `S1-690454716` | `S3-786529773` | `shiva software private limited` | `ச வ ச ஃப ட வ ர ப ர வ ட ல ம ட ட` | `india` | `india` | Address Variation (Minor Street/Zip Difference) |
| `S1-56128179` | `S2-659562652` | `kozhikode innovation corp` | `kozhikode innovation corporation` | `india` | `india` | Shared Name Token / Prefix Match |
| `S1-466792171` | `S3-441459129` | `apc trading limited` | `apc limited center` | `india` | `india` | Shared Name Token / Prefix Match |
| `S1-416433131` | `S2-698357646` | `oasis india services private limited` | `oasis india sgcris private limited` | `india` | `india` | Shared Name Token / Prefix Match |
| `S1-590406770` | `S2-438888115` | `sanroman heritage yhn llc` | `sanroman herieg yhn llc` | `us` | `us` | Shared Name Token / Prefix Match |
| `S1-540582298` | `S2-544651453` | `good technologies private limited` | `க ட ட க ன லஜ ஸ ப ர வ ட ல ம ட ட` | `india` | `india` | Address Variation (Minor Street/Zip Difference) |
| `S1-462292422` | `S3-477178521` | `smith yhn` | `snihm yhn` | `us` | `us` | Shared Name Token / Prefix Match |
| `S1-982168385` | `S3-586856202` | `primary care health inc` | `services primary care inc` | `us` | `us` | Shared Name Token / Prefix Match |
| `S1-394069712` | `S2-184969862` | `bowling richey sidhu alliance inc` | `bowling richey shagidu alliance inc` | `us` | `us` | Shared Name Token / Prefix Match |
| `S1-511868258` | `S2-336845367` | `sidhant sons private limited` | `sidhant sons private` | `india` | `india` | Sub-Name / Missing Tokens (e.g. Inc vs Full) |
| `S1-620913899` | `S3-565191189` | `platinum materials studios` | `platinum studios center` | `us` | `us` | Shared Name Token / Prefix Match |
| `S1-629708529` | `S3-436018185` | `hotel india softtech private limited` | `private h0tel india softtech limited` | `india` | `india` | Shared Name Token / Prefix Match |
| `S1-514941666` | `S2-121540350` | `certified digital technologies inc` | `certified digital technrogise inc` | `us` | `us` | Shared Name Token / Prefix Match |
| `S1-974221288` | `S3-234409124` | `all guardian` | `the all guardian` | `us` | `us` | Sub-Name / Missing Tokens (e.g. Inc vs Full) |
| `S1-209399040` | `S3-224360282` | `rejoice films private limited` | `rejoice private limited center` | `india` | `india` | Shared Name Token / Prefix Match |
| `S1-951862076` | `S3-889989005` | `new logistics pvt ltd` | `ನ ಯ ಲ ಜ ಸ ಟ ಕ ಸ ಪ ರ ವ ಟ ಲ ಮ ಟ ಡ` | `india` | `india` | Address Variation (Minor Street/Zip Difference) |
| `S1-790029773` | `S3-800073909` | `rrs solutions private limited` | `private rrs solutions limited` | `india` | `india` | Shared Name Token / Prefix Match |
| `S1-38767838` | `S2-690434596` | `madelina mancini delta tokyo inc` | `mancini madelina delta inc services` | `us` | `us` | Shared Name Token / Prefix Match |
| `S1-467366747` | `S3-566761962` | `highsmith rocky pelican` | `highsmith rocky peliccan` | `us` | `us` | Shared Name Token / Prefix Match |
| `S1-770103750` | `S3-606500552` | `golden tavern` | `golden tavern services` | `us` | `us` | Sub-Name / Missing Tokens (e.g. Inc vs Full) |
| `S1-673689995` | `S2-339978200` | `lulu consultancy private limited` | `luluconsultancy com` | `india` | `india` | Address Variation (Minor Street/Zip Difference) |
| `S1-503890283` | `S3-306418849` | `dental medicine of barrington` | `dental medicine bbfarigtmn of` | `us` | `us` | Shared Name Token / Prefix Match |
| `S1-166468538` | `S2-282681309` | `fast buildways private limited` | `fast buildways limited private` | `india` | `india` | Shared Name Token / Prefix Match |
| `S1-732914496` | `S2-18613560` | `eastern midwest enterprises` | `eastern midwest` | `us` | `us` | Sub-Name / Missing Tokens (e.g. Inc vs Full) |
| `S1-772561727` | `S2-503070331` | `smart producer limited` | `ସ ମ ର ଟ ପ ର ଡ ୟ ସର ଲ ମ ଟ ଡ` | `india` | `india` | Address Variation (Minor Street/Zip Difference) |
| `S1-757532068` | `S2-43474382` | `my business limited` | `म य ब जन स ल म ट ड` | `india` | `india` | Address Variation (Minor Street/Zip Difference) |
| `S1-600104061` | `S3-879755386` | `wellbeing advisors private limited` | `wellbeingprivateadvisors com` | `india` | `india` | Address Variation (Minor Street/Zip Difference) |
| `S1-11360750` | `S2-72838847` | `laxmi energy private limited` | `लक ष म एनर ज प र इव ट ल म ट ड` | `india` | `india` | Address Variation (Minor Street/Zip Difference) |
| `S1-244672372` | `S3-790262187` | `silver unique it pvt ltd` | `स ल वर य न क आईट प र ल` | `india` | `india` | Address Variation (Minor Street/Zip Difference) |
| `S1-951879843` | `S3-240376969` | `real my impex llp` | `र यल म य इम प क स एलएलप` | `india` | `india` | Address Variation (Minor Street/Zip Difference) |
| `S1-538252181` | `S2-342995822` | `future compost india private limited` | `private future compost rhra i limited` | `india` | `india` | Shared Name Token / Prefix Match |
| `S1-873586746` | `S3-866021894` | `ugj devi pvt ltd` | `ugj ugj devi pvt ltd` | `india` | `india` | Shared Name Token / Prefix Match |
| `S1-220573645` | `S3-688246347` | `sara india technologies private limited` | `sara india techno1ogies private limited` | `india` | `india` | Shared Name Token / Prefix Match |
| `S1-532211364` | `S3-776197056` | `tamil lifecare llp` | `tamil lifecare l l p` | `india` | `india` | Shared Name Token / Prefix Match |
| `S1-279180172` | `S3-844920353` | `franklin county wildlife institute` | `frank1in county wifflfe institute` | `us` | `us` | Shared Name Token / Prefix Match |
