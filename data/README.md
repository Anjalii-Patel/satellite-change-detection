# OSCD Data Summary

The Onera Satellite Change Detection dataset consists of 24 pairs of multispectral Sentinel-2 satellite images. Note that OSCD is small and its labels cover **urban change only**. Vegetation phenology and other non-urban changes are typically not labeled as change.

| Scene ID | Split | Shape | Dates (t1, t2) |
|---|---|---|---|
| abudhabi | Train | 785x799 | date_1: 20160120, date_2: 20180328 |
| aguasclaras | Train | 525x471 | date_1: 20150916, date_2: 20171015 |
| beihai | Train | 772x902 | date_1: 20161209, date_2: 20180309 |
| beirut | Train | 1070x1180 | date_1: 20150820, date_2: 20171003 |
| bercy | Train | 360x395 | date_1: 20161130, date_2: 20170829 |
| bordeaux | Train | 461x517 | date_1: 20160504, date_2: 20171026 |
| brasilia | Test | 469x433 | date_1: 20150916, date_2: 20171017 |
| chongqing | Test | 544x730 | date_1: 20170414, date_2: 20180402 |
| cupertino | Train | 788x1015 | date_1: 20150918, date_2: 20180326 |
| dubai | Test | 634x774 | date_1: 20151211, date_2: 20180330 |
| hongkong | Train | 540x695 | date_1: 20160927, date_2: 20180323 |
| lasvegas | Test | 716x824 | date_1: 20150820, date_2: 20180205 |
| milano | Test | 558x545 | date_1: 20161228, date_2: 20180122 |
| montpellier | Test | 451x426 | date_1: 20150812, date_2: 20171030 |
| mumbai | Train | 557x858 | date_1: 20151130, date_2: 20180319 |
| nantes | Train | 582x522 | date_1: 20150821, date_2: 20171014 |
| norcia | Test | 385x241 | date_1: 20150711, date_2: 20171018 |
| paris | Train | 390x408 | date_1: 20161130, date_2: 20171107 |
| pisa | Train | 718x776 | date_1: 20150704, date_2: 20180211 |
| rennes | Train | 563x339 | date_1: 20150821, date_2: 20170621 |
| rio | Test | 426x353 | date_1: 20160424, date_2: 20171011 |
| saclay_e | Train | 688x639 | date_1: 20160315, date_2: 20170829 |
| saclay_w | Test | 688x639 | date_1: 20160315, date_2: 20170829 |
| valencia | Test | 476x458 | date_1: 20160730, date_2: 20171107 |


## Label Details & Class Imbalance
Labels are provided in .tif format where 1 indicates NO CHANGE and 2 indicates CHANGE. The overall change pixel fraction across the training set is **2.29%**, demonstrating a severe class imbalance.

## Provenance & License
The OSCD dataset uses modified Copernicus data. Original Copernicus Sentinel Data is available from the European Space Agency. The change maps (labels) are released under Creative-Commons BY-NC-SA.
