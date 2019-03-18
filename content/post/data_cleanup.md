---
title: "Data Cleanup"
author: ''
date: "January 12, 2019"
output:
  html_document:
    keep_md: true
    df_print: paged
  pdf_document: default
---



### 1. Preperation

Read in the data:



### 2. Data Cleanup

Let's take a first look at the data:


```
## 'data.frame':	100000 obs. of  19 variables:
##  $ Tier               : Factor w/ 5 levels "Bronze","Diamond",..: 5 4 4 5 5 4 1 5 4 5 ...
##  $ Year               : int  2013 2012 2013 2016 2016 2014 2014 2012 2012 2014 ...
##  $ Insured_First      : Factor w/ 1921 levels "aaden","aadhya",..: 1547 1266 361 801 598 154 695 1362 146 1248 ...
##  $ Insured_Last       : Factor w/ 1000 levels "abbott","acevedo",..: 797 219 479 757 735 365 943 491 535 848 ...
##  $ Territory          : Factor w/ 6 levels "101","102","103",..: 6 6 4 6 6 4 6 6 6 5 ...
##  $ Deductible         : int  500 250 500 500 500 1000 500 500 500 500 ...
##  $ Limit              : int  100000 100000 25000 100000 100000 300000 300000 300000 100000 300000 ...
##  $ Driver_Experience  : Factor w/ 4 levels "High","Low","Medium",..: 3 1 1 1 1 3 2 3 4 4 ...
##  $ Vehicle_Type       : Factor w/ 7 levels "","Car","PICKUP",..: 3 2 2 7 7 6 5 3 5 3 ...
##  $ Prior_Claim        : Factor w/ 3 levels "","No","Yes": 3 2 2 2 2 2 2 2 2 3 ...
##  $ Tenure             : Factor w/ 5 levels "10+","43102",..: 4 4 4 4 4 4 3 4 3 4 ...
##  $ Number_Vehicles    : int  2 2 2 2 2 2 2 4 2 2 ...
##  $ Safe_Driving_Course: Factor w/ 15 levels "No","SYS ERR 0",..: 1 1 15 14 1 15 12 1 1 14 ...
##  $ FB_Friends         : Factor w/ 258 levels "1","10","100",..: 35 258 258 200 195 106 236 158 258 175 ...
##  $ Vehicle_Years      : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ Bi_Indem           : num  0 0 0 0 0 ...
##  $ Bi_Alae            : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ Pd_Indem           : num  0 0 0 1814 0 ...
##  $ Pd_Alae            : num  0 0 0 0 0 0.4 0 0 0 0 ...
```

```
##        Tier            Year      Insured_First     Insured_Last  
##  Bronze  :15280   Min.   :2012   noah   :  793   smith   : 2371  
##  Diamond :11197   1st Qu.:2013   emma   :  780   johnson : 1958  
##  Gold    : 4466   Median :2014   james  :  778   williams: 1621  
##  Platinum:19824   Mean   :2014   liam   :  777   jones   : 1426  
##  Silver  :49233   3rd Qu.:2015   olivia :  773   brown   : 1413  
##                   Max.   :2016   amelia :  772   davis   : 1086  
##                                  (Other):95327   (Other) :90125  
##   Territory       Deductible         Limit        Driver_Experience
##  101   : 2729   Min.   : 250.0   Min.   : 25000   High     :39678  
##  102   :  872   1st Qu.: 250.0   1st Qu.: 25000   Low      :10682  
##  103   :  865   Median : 500.0   Median :100000   Medium   :29856  
##  Metro :23519   Mean   : 498.2   Mean   :173105   Very High:19784  
##  Rural :18657   3rd Qu.: 500.0   3rd Qu.:300000                    
##  Suburb:53358   Max.   :1000.0   Max.   :300000                    
##                                                                    
##   Vehicle_Type   Prior_Claim   Tenure      Number_Vehicles
##         : 1170      : 4466   10+  :14464   Min.   :1.000  
##  Car    :26458   No :84144   43102:10101   1st Qu.:1.000  
##  PICKUP :13096   Yes:11390   43163:20186   Median :2.000  
##  SEDAN  :23520               43230:49822   Mean   :2.052  
##  Truck  :11502               New  : 5427   3rd Qu.:2.000  
##  Utility:11637                             Max.   :5.000  
##  UTILITY:12617                                            
##  Safe_Driving_Course   FB_Friends    Vehicle_Years      Bi_Indem       
##  No        :44053    Na     :32411   Min.   :0.000   Min.   : -1000.0  
##  Yes       :29824    100    :11212   1st Qu.:1.000   1st Qu.:     0.0  
##  Unknown   :14705    5      : 4055   Median :1.000   Median :     0.0  
##  SYS ERR 6 : 1042    3      : 3664   Mean   :1.094   Mean   :   431.8  
##  SYS ERR 10: 1031    4      : 3585   3rd Qu.:1.000   3rd Qu.:     0.0  
##  SYS ERR 8 : 1017    2      : 3363   Max.   :5.000   Max.   :999999.0  
##  (Other)   : 8328    (Other):41710                                     
##     Bi_Alae            Pd_Indem          Pd_Alae       
##  Min.   :    0.00   Min.   :    0.0   Min.   :  0.000  
##  1st Qu.:    0.00   1st Qu.:    0.0   1st Qu.:  0.000  
##  Median :    0.00   Median :    0.0   Median :  0.000  
##  Mean   :   17.11   Mean   :  184.8   Mean   :  1.759  
##  3rd Qu.:    0.00   3rd Qu.:    0.0   3rd Qu.:  0.000  
##  Max.   :19978.00   Max.   :10076.0   Max.   :996.000  
## 
```

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["Tier"],"name":[1],"type":["fctr"],"align":["left"]},{"label":["Year"],"name":[2],"type":["int"],"align":["right"]},{"label":["Insured_First"],"name":[3],"type":["fctr"],"align":["left"]},{"label":["Insured_Last"],"name":[4],"type":["fctr"],"align":["left"]},{"label":["Territory"],"name":[5],"type":["fctr"],"align":["left"]},{"label":["Deductible"],"name":[6],"type":["int"],"align":["right"]},{"label":["Limit"],"name":[7],"type":["int"],"align":["right"]},{"label":["Driver_Experience"],"name":[8],"type":["fctr"],"align":["left"]},{"label":["Vehicle_Type"],"name":[9],"type":["fctr"],"align":["left"]},{"label":["Prior_Claim"],"name":[10],"type":["fctr"],"align":["left"]},{"label":["Tenure"],"name":[11],"type":["fctr"],"align":["left"]},{"label":["Number_Vehicles"],"name":[12],"type":["int"],"align":["right"]},{"label":["Safe_Driving_Course"],"name":[13],"type":["fctr"],"align":["left"]},{"label":["FB_Friends"],"name":[14],"type":["fctr"],"align":["left"]},{"label":["Vehicle_Years"],"name":[15],"type":["int"],"align":["right"]},{"label":["Bi_Indem"],"name":[16],"type":["dbl"],"align":["right"]},{"label":["Bi_Alae"],"name":[17],"type":["dbl"],"align":["right"]},{"label":["Pd_Indem"],"name":[18],"type":["dbl"],"align":["right"]},{"label":["Pd_Alae"],"name":[19],"type":["dbl"],"align":["right"]}],"data":[{"1":"Silver","2":"2013","3":"phoebe","4":"ross","5":"Suburb","6":"500","7":"100000","8":"Medium","9":"PICKUP","10":"Yes","11":"43230","12":"2","13":"No","14":"13","15":"1","16":"0","17":"0","18":"0.0","19":"0.0","_rn_":"1"},{"1":"Platinum","2":"2012","3":"lucille","4":"dale","5":"Suburb","6":"250","7":"100000","8":"High","9":"Car","10":"No","11":"43230","12":"2","13":"No","14":"Na","15":"1","16":"0","17":"0","18":"0.0","19":"0.0","_rn_":"2"},{"1":"Platinum","2":"2013","3":"cadence","4":"johnson","5":"Metro","6":"500","7":"25000","8":"High","9":"Car","10":"No","11":"43230","12":"2","13":"Yes","14":"Na","15":"1","16":"0","17":"0","18":"0.0","19":"0.0","_rn_":"3"},{"1":"Silver","2":"2016","3":"hattie","4":"reese","5":"Suburb","6":"500","7":"100000","8":"High","9":"UTILITY","10":"No","11":"43230","12":"2","13":"Unknown","14":"5","15":"1","16":"0","17":"0","18":"1814.0","19":"0.0","_rn_":"4"},{"1":"Silver","2":"2016","3":"elijah","4":"potter","5":"Suburb","6":"500","7":"100000","8":"High","9":"UTILITY","10":"No","11":"43230","12":"2","13":"No","14":"46","15":"1","16":"0","17":"0","18":"0.0","19":"0.0","_rn_":"5"},{"1":"Platinum","2":"2014","3":"andrew","4":"griffin","5":"Metro","6":"1000","7":"300000","8":"Medium","9":"Utility","10":"No","11":"43230","12":"2","13":"Yes","14":"2","15":"1","16":"0","17":"0","18":"2.7","19":"0.4","_rn_":"6"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>

We see that Territory has levels other than "Metro", "Suburb" and "Rural". Vehicle_Type has levels other than "Car", "Truck" and "Utility". It also has missing values. Prior_Claim has missing values as well. Tenure has levels different from the directory ("New", "1-2", "3-4", "5-10" and "10+"). Safe_Driving_Course has levels other than "Yes", "No" and "Unknown". FB_Friends has missing values. Vehicle_Years has values other than 1. Bi_Indem has negative values. Also some of the variable types are wrong.

#### 2.1 cleanup Vehicle_Type


```
##           
##                    Car PICKUP SEDAN Truck Utility UTILITY  <NA>
##   Bronze       0  8690      0     0  3242    3348       0     0
##   Diamond      0  5289      0     0  2999    2909       0     0
##   Gold         0  2948      0     0   693     825       0     0
##   Platinum  1170  9531      0     0  4568    4555       0     0
##   Silver       0     0  13096 23520     0       0   12617     0
##   <NA>         0     0      0     0     0       0       0     0
```

From the frequency table above, we see that all the missing values of Vehicle_Type belongs to Platinum. Silver has different notation for Vehicle_Type. Therefore, we will change those Vehicle_Type accordingly.


```r
data = data %>%
  mutate(Vehicle_Type = case_when(Vehicle_Type=="SEDAN" ~ "Car",
                                  Vehicle_Type=="PICKUP" ~ "Truck",
                                  Vehicle_Type=="UTILITY" ~ "Utility",
                                  Vehicle_Type=="" ~ NA_character_,
                                  TRUE ~ as.character(Vehicle_Type)))
```

#### 2.2 cleanup Territory


```
##           
##              101   102   103 Metro Rural Suburb  <NA>
##   Bronze       0     0     0  6184  2962   6134     0
##   Diamond      0     0     0  2340  1031   7826     0
##   Gold      2729   872   865     0     0      0     0
##   Platinum     0     0     0  5039  4947   9838     0
##   Silver       0     0     0  9956  9717  29560     0
##   <NA>         0     0     0     0     0      0     0
```

From the frequency table above, we see that Gold has different notation for Territory. Let's look at the frequency ratio of car to truck, truck to truck and utility to truck for different Tier and Territory combinations to see if there is any pattern.

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":["Tier"],"name":[1],"type":["fctr"],"align":["left"]},{"label":["Territory"],"name":[2],"type":["fctr"],"align":["left"]},{"label":["Car_to_Truck_ratio"],"name":[3],"type":["dbl"],"align":["right"]},{"label":["Truck_to_Truck_ratio"],"name":[4],"type":["dbl"],"align":["right"]},{"label":["Utility_to_Truck_ratio"],"name":[5],"type":["dbl"],"align":["right"]}],"data":[{"1":"Bronze","2":"Metro","3":"9.522158","4":"1","5":"1.3930636"},{"1":"Bronze","2":"Rural","3":"2.106796","4":"1","5":"1.0013870"},{"1":"Bronze","2":"Suburb","3":"1.113387","4":"1","5":"0.9505495"},{"1":"Diamond","2":"Metro","3":"10.118919","4":"1","5":"1.5297297"},{"1":"Diamond","2":"Rural","3":"2.036437","4":"1","5":"1.1376518"},{"1":"Diamond","2":"Suburb","3":"1.135177","4":"1","5":"0.9135177"},{"1":"Gold","2":"101","3":"9.780269","4":"1","5":"1.4573991"},{"1":"Gold","2":"102","3":"1.161172","4":"1","5":"1.0329670"},{"1":"Gold","2":"103","3":"2.284264","4":"1","5":"1.1065990"},{"1":"Platinum","2":"Metro","3":"10.435616","4":"1","5":"1.5616438"},{"1":"Platinum","2":"Rural","3":"1.992399","4":"1","5":"0.9518581"},{"1":"Platinum","2":"Suburb","3":"1.113945","4":"1","5":"0.9466711"},{"1":"Silver","2":"Metro","3":"10.323454","4":"1","5":"1.5064433"},{"1":"Silver","2":"Rural","3":"1.990965","4":"1","5":"0.9995893"},{"1":"Silver","2":"Suburb","3":"1.078503","4":"1","5":"0.9118867"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>

We see that the three ratios are very close within the same Territory for different Tier. The three ratios for 101 is very close to Metro. The three ratios for 102 is very close to Suburb The three ratios for 103 is very close to Rural. Therefore, we will fix Territory accordingly.


```r
data = data %>%
  mutate(Territory = case_when(Territory=="101" & Tier=="Gold" ~ "Metro",
                               Territory=="102" & Tier=="Gold" ~ "Suburb",
                               Territory=="103" & Tier=="Gold" ~ "Rural",
                               TRUE ~ as.character(Territory)))
```

#### 2.3 cleanup Tenure

Tenure can only take value "New", "1-2", "3-4", "5-10" and "10+". However, Excel automatically recognizes them as dates so they become "New", "2-Jan", "4-Mar", "10-May" and "10+" respectively. Data is imported to R after converting all columns to text in Excel, so these levels becomes "New", "43102", "43163", "43230" and "10+" respectively during the conversion. Therefore, we will fix Tenure accordingly.


```r
data = data %>%
  mutate(Tenure = case_when(Tenure=="New" ~ "New",
                   Tenure=="43102" ~ "1-2",
                   Tenure=="43163" ~ "3-4",
                   Tenure=="43230" ~ "5-10",
                   Tenure=="10+" ~ "10+",
                   TRUE ~ as.character(Tenure)))
```

#### 2.4 cleanup Safe_Driving_Course

Safe_Driving_Course has values such as "SYS ERR 0", "SYS ERR 1"..., so it is reasonable to consider them as missing values. Therefore, we change them to NA.


```r
data = data %>%
  mutate(Safe_Driving_Course = case_when(!(Safe_Driving_Course %in% c("Yes", "No", "Unknown")) ~ NA_character_,
                                         TRUE ~ as.character(Safe_Driving_Course)))
```


```
##           
##                   No   Unknown       Yes      <NA>
##   Bronze   0.0000000 0.0000000 0.8878272 0.1121728
##   Diamond  0.5238010 0.1783513 0.1806734 0.1171742
##   Gold     0.6206897 0.1679355 0.0911330 0.1202418
##   Platinum 0.5763216 0.1676755 0.1411421 0.1148608
##   Silver   0.4872951 0.1753702 0.2240367 0.1132980
##   <NA>
```

From the proportion table ablove, we see that Bronze has 0 No and Unknown, but its Yes proportion is significantly larger than other Tier.

#### 2.5 cleanup FB_Friends


```r
par(mfrow=c(2,3))
hist(as.numeric(as.character(data$FB_Friends[data$Tier=="Bronze"])),
     main="Histogram for Bronze Tier",xlab="FB_Friends")
hist(as.numeric(as.character(data$FB_Friends[data$Tier=="Silver"])),
     main="Histogram for Silver Tier",xlab="FB_Friends")
hist(as.numeric(as.character(data$FB_Friends[data$Tier=="Gold"])),
     main="Histogram for Gold Tier",xlab="FB_Friends")
hist(as.numeric(as.character(data$FB_Friends[data$Tier=="Platinum"])),
     main="Histogram for Platinum Tier",xlab="FB_Friends")
hist(as.numeric(as.character(data$FB_Friends[data$Tier=="Diamond"])),
     main="Histogram for Diamond Tier",xlab="FB_Friends")
```

![](data_cleanup_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

We see that tiers other than Diamond has similar distribution of FB_Friends, while FB_Friends is 100 for all customers in Diamond. It does not make sense for an insurance company to only find people with 100 Facebook friends to insure, so the FB_Friends in Diamond is not the true value (probably the default value). Therefore, we change them to NA.


```r
data = data %>%
  mutate(FB_Friends = case_when(FB_Friends=="NA" | Tier=="Diamond" ~ NA_character_,
                                TRUE ~ as.character(FB_Friends)))
```

#### 2.6 cleanup Prior_Claim

We change blanks in Prior_Claim to NA.


```r
data = data %>%
  mutate(Prior_Claim = case_when(Prior_Claim=="" ~ NA_character_,
                                 TRUE ~ as.character(Prior_Claim)))
```

#### 2.7 cleanup Vehicle_Years


```
##           
##                0     1     2     3     4     5  <NA>
##   Bronze       0 15280     0     0     0     0     0
##   Diamond    963  3355  4721  1282   462   414     0
##   Gold         0  4466     0     0     0     0     0
##   Platinum     0 19824     0     0     0     0     0
##   Silver       0 49233     0     0     0     0     0
##   <NA>         0     0     0     0     0     0     0
```

From the frequency table above, we see that only Diamond has values other than 1. Let's take a look at row with Vehicle_Years = 0.

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["Tier"],"name":[1],"type":["fctr"],"align":["left"]},{"label":["Year"],"name":[2],"type":["int"],"align":["right"]},{"label":["Insured_First"],"name":[3],"type":["fctr"],"align":["left"]},{"label":["Insured_Last"],"name":[4],"type":["fctr"],"align":["left"]},{"label":["Territory"],"name":[5],"type":["chr"],"align":["left"]},{"label":["Deductible"],"name":[6],"type":["int"],"align":["right"]},{"label":["Limit"],"name":[7],"type":["int"],"align":["right"]},{"label":["Driver_Experience"],"name":[8],"type":["fctr"],"align":["left"]},{"label":["Vehicle_Type"],"name":[9],"type":["chr"],"align":["left"]},{"label":["Prior_Claim"],"name":[10],"type":["chr"],"align":["left"]},{"label":["Tenure"],"name":[11],"type":["chr"],"align":["left"]},{"label":["Number_Vehicles"],"name":[12],"type":["int"],"align":["right"]},{"label":["Safe_Driving_Course"],"name":[13],"type":["chr"],"align":["left"]},{"label":["FB_Friends"],"name":[14],"type":["chr"],"align":["left"]},{"label":["Vehicle_Years"],"name":[15],"type":["int"],"align":["right"]},{"label":["Bi_Indem"],"name":[16],"type":["dbl"],"align":["right"]},{"label":["Bi_Alae"],"name":[17],"type":["dbl"],"align":["right"]},{"label":["Pd_Indem"],"name":[18],"type":["dbl"],"align":["right"]},{"label":["Pd_Alae"],"name":[19],"type":["dbl"],"align":["right"]}],"data":[{"1":"Diamond","2":"2012","3":"abigail","4":"bell","5":"Suburb","6":"500","7":"100000","8":"High","9":"Truck","10":"Yes","11":"New","12":"2","13":"Yes","14":"NA","15":"0","16":"7534","17":"1730","18":"0","19":"0","_rn_":"1"},{"1":"Diamond","2":"2012","3":"abigail","4":"bell","5":"Suburb","6":"500","7":"100000","8":"High","9":"Truck","10":"Yes","11":"New","12":"2","13":"Yes","14":"NA","15":"1","16":"0","17":"0","18":"0","19":"0","_rn_":"2"},{"1":"Diamond","2":"2012","3":"abigail","4":"haynes","5":"Suburb","6":"500","7":"100000","8":"High","9":"Utility","10":"No","11":"5-10","12":"2","13":"Yes","14":"NA","15":"0","16":"0","17":"0","18":"3034","19":"356","_rn_":"3"},{"1":"Diamond","2":"2012","3":"abigail","4":"haynes","5":"Suburb","6":"500","7":"100000","8":"High","9":"Utility","10":"No","11":"5-10","12":"2","13":"Yes","14":"NA","15":"1","16":"0","17":"0","18":"0","19":"0","_rn_":"4"},{"1":"Diamond","2":"2012","3":"abigail","4":"moore","5":"Suburb","6":"500","7":"25000","8":"High","9":"Utility","10":"No","11":"10+","12":"2","13":"Yes","14":"NA","15":"0","16":"0","17":"0","18":"1389","19":"0","_rn_":"5"},{"1":"Diamond","2":"2012","3":"abigail","4":"moore","5":"Suburb","6":"500","7":"25000","8":"High","9":"Utility","10":"No","11":"10+","12":"2","13":"Yes","14":"NA","15":"1","16":"0","17":"0","18":"0","19":"0","_rn_":"6"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>

We found that every row in Diamond with Vehicle_Years = 0 has claim amount > 0. And for every one of these rows, there is an other row in Diamond with Vehicle_Years = 1 and claim amount = 0. One can see this from the rearranged data above.


```
##       
##            1     2     3     4     5  <NA>
##   1    25115     0     0     0     0     0
##   2    49173  4721     0     0     0     0
##   3    10294     0  1282     0     0     0
##   4     3469     0     0   462     0     0
##   5     3144     0     0     0   414     0
##   <NA>     0     0     0     0     0     0
```

After removing the rows mentioned above, we found that Number_Vehicles = Vehicle_Years for the rest rows in Diamond. The above frequency table has Number_Vehicles as rows and Vehicle_Years as columns. Therefore, we fix Vehicle_Years accordingly.


```r
data =
  data %>%
  filter(Tier=="Diamond" & Vehicle_Years==0) %>%
  right_join(data %>%
               filter(!(Tier=="Diamond" & Vehicle_Years==0)),
             by = c("Tier","Year","Insured_First","Insured_Last","Territory",
                    "Deductible","Limit","Driver_Experience","Vehicle_Type","Prior_Claim",
                    "Tenure","Number_Vehicles","Safe_Driving_Course","FB_Friends")) %>%
  mutate(Vehicle_Years = 1,
         Bi_Indem = case_when(is.na(Bi_Indem.x) ~ Bi_Indem.y,
                              TRUE ~ Bi_Indem.x),
         Bi_Alae = case_when(is.na(Bi_Alae.x) ~ Bi_Alae.y,
                              TRUE ~ Bi_Alae.x),
         Pd_Indem = case_when(is.na(Pd_Indem.x) ~ Pd_Indem.y,
                              TRUE ~ Pd_Indem.x),
         Pd_Alae = case_when(is.na(Pd_Alae.x) ~ Pd_Alae.y,
                              TRUE ~ Pd_Alae.x)) %>%
  select(Tier,Year,Insured_First,Insured_Last,Territory,Deductible,Limit,Driver_Experience,
         Vehicle_Type,Prior_Claim,Tenure,Number_Vehicles,Safe_Driving_Course,FB_Friends,
         Vehicle_Years,Bi_Indem,Bi_Alae,Pd_Indem,Pd_Alae)
```

#### 2.8 cleanup Bi_Indem

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":[""],"name":["_rn_"],"type":[""],"align":["left"]},{"label":["Tier"],"name":[1],"type":["fctr"],"align":["left"]},{"label":["Year"],"name":[2],"type":["int"],"align":["right"]},{"label":["Insured_First"],"name":[3],"type":["fctr"],"align":["left"]},{"label":["Insured_Last"],"name":[4],"type":["fctr"],"align":["left"]},{"label":["Territory"],"name":[5],"type":["chr"],"align":["left"]},{"label":["Deductible"],"name":[6],"type":["int"],"align":["right"]},{"label":["Limit"],"name":[7],"type":["int"],"align":["right"]},{"label":["Driver_Experience"],"name":[8],"type":["fctr"],"align":["left"]},{"label":["Vehicle_Type"],"name":[9],"type":["chr"],"align":["left"]},{"label":["Prior_Claim"],"name":[10],"type":["chr"],"align":["left"]},{"label":["Tenure"],"name":[11],"type":["chr"],"align":["left"]},{"label":["Number_Vehicles"],"name":[12],"type":["int"],"align":["right"]},{"label":["Safe_Driving_Course"],"name":[13],"type":["chr"],"align":["left"]},{"label":["FB_Friends"],"name":[14],"type":["chr"],"align":["left"]},{"label":["Vehicle_Years"],"name":[15],"type":["dbl"],"align":["right"]},{"label":["Bi_Indem"],"name":[16],"type":["dbl"],"align":["right"]},{"label":["Bi_Alae"],"name":[17],"type":["dbl"],"align":["right"]},{"label":["Pd_Indem"],"name":[18],"type":["dbl"],"align":["right"]},{"label":["Pd_Alae"],"name":[19],"type":["dbl"],"align":["right"]}],"data":[{"1":"Bronze","2":"2012","3":"aaron","4":"nunez","5":"Suburb","6":"500","7":"25000","8":"Low","9":"Car","10":"No","11":"3-4","12":"2","13":"Yes","14":"9","15":"1","16":"-500","17":"0","18":"0","19":"0","_rn_":"1"},{"1":"Bronze","2":"2012","3":"aaron","4":"nunez","5":"Suburb","6":"500","7":"25000","8":"Low","9":"Car","10":"No","11":"3-4","12":"2","13":"Yes","14":"9","15":"1","16":"2393","17":"184","18":"0","19":"0","_rn_":"2"},{"1":"Bronze","2":"2012","3":"abigail","4":"henderson","5":"Suburb","6":"250","7":"100000","8":"High","9":"Truck","10":"No","11":"5-10","12":"2","13":"NA","14":"Na","15":"1","16":"-250","17":"0","18":"0","19":"0","_rn_":"3"},{"1":"Bronze","2":"2012","3":"abigail","4":"henderson","5":"Suburb","6":"250","7":"100000","8":"High","9":"Truck","10":"No","11":"5-10","12":"2","13":"NA","14":"Na","15":"1","16":"2550","17":"0","18":"0","19":"0","_rn_":"4"},{"1":"Bronze","2":"2012","3":"addison","4":"jones","5":"Suburb","6":"500","7":"100000","8":"High","9":"Truck","10":"No","11":"5-10","12":"3","13":"Yes","14":"10","15":"1","16":"-500","17":"0","18":"0","19":"0","_rn_":"5"},{"1":"Bronze","2":"2012","3":"addison","4":"jones","5":"Suburb","6":"500","7":"100000","8":"High","9":"Truck","10":"No","11":"5-10","12":"3","13":"Yes","14":"10","15":"1","16":"2186","17":"0","18":"0","19":"0","_rn_":"6"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>

There are random rows in Bronze with negative Bi_Indem. And these Bi_Indem are the same as the negative value of the Deductible on the same row. One can see this from the rearranged data above. Therefore, we fix Bi_Indem by deleting the redundant rows.


```r
data = data %>% filter(!(Bi_Indem < 0 & Tier == "Bronze"))
```

#### 2.9 Duplicate rows

<div data-pagedtable="false">
  <script data-pagedtable-source type="application/json">
{"columns":[{"label":["Tier"],"name":[1],"type":["fctr"],"align":["left"]},{"label":["Year"],"name":[2],"type":["int"],"align":["right"]},{"label":["Insured_First"],"name":[3],"type":["fctr"],"align":["left"]},{"label":["Insured_Last"],"name":[4],"type":["fctr"],"align":["left"]},{"label":["Territory"],"name":[5],"type":["chr"],"align":["left"]},{"label":["Deductible"],"name":[6],"type":["int"],"align":["right"]},{"label":["Limit"],"name":[7],"type":["int"],"align":["right"]},{"label":["Driver_Experience"],"name":[8],"type":["fctr"],"align":["left"]},{"label":["Vehicle_Type"],"name":[9],"type":["chr"],"align":["left"]},{"label":["Prior_Claim"],"name":[10],"type":["chr"],"align":["left"]},{"label":["Tenure"],"name":[11],"type":["chr"],"align":["left"]},{"label":["Number_Vehicles"],"name":[12],"type":["int"],"align":["right"]},{"label":["Safe_Driving_Course"],"name":[13],"type":["chr"],"align":["left"]},{"label":["FB_Friends"],"name":[14],"type":["chr"],"align":["left"]},{"label":["Vehicle_Years"],"name":[15],"type":["dbl"],"align":["right"]},{"label":["Bi_Indem"],"name":[16],"type":["dbl"],"align":["right"]},{"label":["Bi_Alae"],"name":[17],"type":["dbl"],"align":["right"]},{"label":["Pd_Indem"],"name":[18],"type":["dbl"],"align":["right"]},{"label":["Pd_Alae"],"name":[19],"type":["dbl"],"align":["right"]}],"data":[{"1":"Platinum","2":"2012","3":"aiden","4":"russell","5":"Suburb","6":"500","7":"300000","8":"High","9":"NA","10":"No","11":"5-10","12":"1","13":"No","14":"19","15":"1","16":"0","17":"0","18":"0","19":"0"},{"1":"Platinum","2":"2012","3":"aiden","4":"russell","5":"Suburb","6":"500","7":"300000","8":"High","9":"NA","10":"No","11":"5-10","12":"1","13":"No","14":"19","15":"1","16":"0","17":"0","18":"0","19":"0"},{"1":"Platinum","2":"2012","3":"albert","4":"diaz","5":"Suburb","6":"500","7":"300000","8":"High","9":"NA","10":"No","11":"3-4","12":"2","13":"Yes","14":"3","15":"1","16":"0","17":"0","18":"0","19":"0"},{"1":"Platinum","2":"2012","3":"albert","4":"diaz","5":"Suburb","6":"500","7":"300000","8":"High","9":"NA","10":"No","11":"3-4","12":"2","13":"Yes","14":"3","15":"1","16":"0","17":"0","18":"0","19":"0"},{"1":"Platinum","2":"2012","3":"alexander","4":"combs","5":"Suburb","6":"250","7":"100000","8":"Medium","9":"NA","10":"No","11":"5-10","12":"2","13":"Yes","14":"Na","15":"1","16":"0","17":"0","18":"0","19":"0"},{"1":"Platinum","2":"2012","3":"alexander","4":"combs","5":"Suburb","6":"250","7":"100000","8":"Medium","9":"NA","10":"No","11":"5-10","12":"2","13":"Yes","14":"Na","15":"1","16":"0","17":"0","18":"0","19":"0"}],"options":{"columns":{"min":{},"max":[10]},"rows":{"min":[10],"max":[10]},"pages":{}}}
  </script>
</div>

```
##           
##                1     2  <NA>
##   Bronze   14923     0     0
##   Diamond  10234     0     0
##   Gold      4466     0     0
##   Platinum 19460   182     0
##   Silver   49233     0     0
##   <NA>         0     0     0
```

There are still 182 duplicate rows as shown in the rearranged data above. The frequency table has the row count as columns. It shows that all the duplicate rows have one extra row and are all in Platinum. Note that all these duplicate rows have missing Vehicle_Type, but not all rows in Platinum with missing Vehicle_Type have duplicate rows.

We did not remove these duplicate rows because: 1. It is still possible that two people have exactly the same value for all the columns. 2. There is no obvious pattern in the way these rows are duplicated. 3. The proportion of duplicate rows are very small (364/98680 = 0.37%). We don't think our prediction will change much even these duplcate rows removed.

#### 2.10 Adding ID

Finally, we add IDs to all the customers in the data. Note that based on the reasoning in part 2.9, each row will represent a unique person.


```r
data = data %>%
  mutate(ID = row_number()) %>%
  select(ID, everything())
```

#### 2.11 Change variable type

Correct variable type.


```r
data = data %>%
  mutate(ID=as.factor(ID), Tier=as.factor(Tier), Year=as.factor(Year),
         Insured_First=as.character(Insured_First), Insured_Last=as.character(Insured_Last),
         Territory=as.factor(Territory), Deductible=as.numeric(Deductible),
         Limit=as.numeric(Limit), Driver_Experience=as.factor(Driver_Experience),
         Vehicle_Type=as.factor(Vehicle_Type), Prior_Claim=as.factor(Prior_Claim),
         Tenure=as.factor(Tenure), Number_Vehicles=as.numeric(Number_Vehicles),
         Safe_Driving_Course=as.factor(Safe_Driving_Course),
         FB_Friends=as.numeric(FB_Friends), Vehicle_Years=as.numeric(Vehicle_Years),
         Bi_Indem=as.numeric(Bi_Indem), Bi_Alae=as.numeric(Bi_Alae), Pd_Indem=as.numeric(Pd_Indem),
         Pd_Alae=as.numeric(Pd_Alae))
```
