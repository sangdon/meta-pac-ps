for YEAR in 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020
do
    wget -nc https://www.cdc.gov/brfss/annual_data/${YEAR}/files/LLCP${YEAR}XPT.zip
    unzip -n LLCP${YEAR}XPT.zip
    mv --backup=existing LLCP${YEAR}.XPT\  LLCP${YEAR}.XPT
    chmod u+rw LLCP${YEAR}.XPT
    chmod g+r LLCP${YEAR}.XPT
    chmod o+r LLCP${YEAR}.XPT
done
