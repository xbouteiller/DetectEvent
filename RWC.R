install.packages("tidyverse")
library(tidyverse)
install.packages("lubridate")
library(lubridate)


############################################################################################################################

setwd("C:/Users/Thobaout/Desktop/Stage/R/Exploited_data")
Data <- read.csv("WATERLOO_15_03_2021_value_TAOF.csv", dec=".")
Area <- read.csv("WATERLOO_12_03_2021_area_TAOF.csv")


#format date 

Data$date_time<-dmy_hms(Data$date_time)
Data$date_time<-dmy_hm(Data$date_time)

#Pivot_longer

Data <- Data %>%
  group_by(date_time) %>% 
  pivot_longer(names_to="Position",
               values_to="weight",
               cols = -c("date_time","Campaign","Temp_A", "RH_A", "Temp_B", "RH_B","Temp_C", "RH_C","Temp_D", "RH_D"),
               values_drop_na = TRUE
  ) 

#Plot de verification

plot <- Data %>%
  ggplot(aes(date_time,weight,color=Position))+geom_line()+geom_point()+ facet_wrap(~Position)+ scale_y_continuous(c(min=0, max =5))


#Filter
Data <- Data %>% filter(weight<8, weight>-2,.preserve = FALSE)
?filter
#Plot de verification

plot <- Data %>%
  ggplot(aes(date_time,weight,color=Position))+geom_line()+geom_point()+ facet_wrap(~Position)+ scale_y_continuous(c(min=0, max =5))

plot
#Mutate Temp

Data <- Data %>%
  mutate(T_C=(Temp_A+Temp_B)/2)

#Mutate RH

Data <- Data %>%
  mutate(RH=(RH_A+RH_B)/2)

#Subset -c(TEMP_A/B/C/D; RH_A/B/C/D)

Data <- Data %>% subset(select=-c(Temp_A,Temp_B,Temp_C,Temp_D,RH_A,RH_B,RH_C,RH_D))

#left_join

Fusion<-left_join(Data,Area, by="Position")

#Mutate RWC

Fusion <- Fusion %>% 
  mutate(RWC=100*((weight-Out.Mass..g.)/(Fresh.Mass..g.-Out.Mass..g.)))

plot <- Fusion %>%
  ggplot(aes(date_time.x,RWC,color=Position))+geom_line()+geom_point()+ facet_wrap(~Position)+ scale_y_continuous(c(min=0, max =100))

plot

Fusion <- Fusion %>% filter(RWC<150, RWC>-30,.preserve = FALSE)
#Subset all pointless data

Fusion <- Fusion %>%  subset(select=-c(date_time.y,Fresh.Mass..g.,Out.Mass..g.,Needle.length..cm.,Needle.width..cm.,Needle.n.,`Mean.needle.Area..cm².`,Location))

#Création Patm

Fusion <- Fusion %>% mutate(Patm=101.325)

#Réarrangement

Fusion <- arrange(Fusion, Position)

#Retrait position

Fusion <- Fusion %>%  subset(select=-c(Position))

#Uniformisation noms colonnes

names(Fusion)[1] <- "date_time"
names(Fusion)[2] <- "campaign"
names(Fusion)[3] <- "weight_g"
names(Fusion)[6] <- "sample_ID"
names(Fusion)[7] <- "Area_m2"
as.numeric(data$Area_m2)
str(Fusion)
############################################################################################################################


#Extraction dataframe

write_excel_csv(Fusion, "C:/Users/Thobaout/Desktop/Stage/Python/WATERLOO_XX_XX_XX_processed_PIHA_HEHE.csv")
