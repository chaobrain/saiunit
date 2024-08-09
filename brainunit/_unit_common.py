# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from ._base import Unit, get_or_create_dimension

__all__ = [
  "metre",
  "meter",
  "kilogram",
  "second",
  "amp",
  "ampere",
  "kelvin",
  "mole",
  "mol",
  "candle",
  "kilogramme",
  "gram",
  "gramme",
  "molar",
  "radian",
  "steradian",
  "hertz",
  "newton",
  "pascal",
  "joule",
  "watt",
  "coulomb",
  "volt",
  "farad",
  "ohm",
  "siemens",
  "weber",
  "tesla",
  "henry",
  "lumen",
  "lux",
  "becquerel",
  "gray",
  "sievert",
  "katal",
  "ametre",
  "cmetre",
  "Zmetre",
  "Pmetre",
  "dmetre",
  "Gmetre",
  "fmetre",
  "hmetre",
  "dametre",
  "mmetre",
  "nmetre",
  "pmetre",
  "umetre",
  "Tmetre",
  "ymetre",
  "Emetre",
  "zmetre",
  "Mmetre",
  "kmetre",
  "Ymetre",
  "ameter",
  "cmeter",
  "Zmeter",
  "Pmeter",
  "dmeter",
  "Gmeter",
  "fmeter",
  "hmeter",
  "dameter",
  "mmeter",
  "nmeter",
  "pmeter",
  "umeter",
  "Tmeter",
  "ymeter",
  "Emeter",
  "zmeter",
  "Mmeter",
  "kmeter",
  "Ymeter",
  "asecond",
  "csecond",
  "Zsecond",
  "Psecond",
  "dsecond",
  "Gsecond",
  "fsecond",
  "hsecond",
  "dasecond",
  "msecond",
  "nsecond",
  "psecond",
  "usecond",
  "Tsecond",
  "ysecond",
  "Esecond",
  "zsecond",
  "Msecond",
  "ksecond",
  "Ysecond",
  "aamp",
  "camp",
  "Zamp",
  "Pamp",
  "damp",
  "Gamp",
  "famp",
  "hamp",
  "daamp",
  "mamp",
  "namp",
  "pamp",
  "uamp",
  "Tamp",
  "yamp",
  "Eamp",
  "zamp",
  "Mamp",
  "kamp",
  "Yamp",
  "aampere",
  "campere",
  "Zampere",
  "Pampere",
  "dampere",
  "Gampere",
  "fampere",
  "hampere",
  "daampere",
  "mampere",
  "nampere",
  "pampere",
  "uampere",
  "Tampere",
  "yampere",
  "Eampere",
  "zampere",
  "Mampere",
  "kampere",
  "Yampere",
  "amole",
  "cmole",
  "Zmole",
  "Pmole",
  "dmole",
  "Gmole",
  "fmole",
  "hmole",
  "damole",
  "mmole",
  "nmole",
  "pmole",
  "umole",
  "Tmole",
  "ymole",
  "Emole",
  "zmole",
  "Mmole",
  "kmole",
  "Ymole",
  "amol",
  "cmol",
  "Zmol",
  "Pmol",
  "dmol",
  "Gmol",
  "fmol",
  "hmol",
  "damol",
  "mmol",
  "nmol",
  "pmol",
  "umol",
  "Tmol",
  "ymol",
  "Emol",
  "zmol",
  "Mmol",
  "kmol",
  "Ymol",
  "acandle",
  "ccandle",
  "Zcandle",
  "Pcandle",
  "dcandle",
  "Gcandle",
  "fcandle",
  "hcandle",
  "dacandle",
  "mcandle",
  "ncandle",
  "pcandle",
  "ucandle",
  "Tcandle",
  "ycandle",
  "Ecandle",
  "zcandle",
  "Mcandle",
  "kcandle",
  "Ycandle",
  "agram",
  "cgram",
  "Zgram",
  "Pgram",
  "dgram",
  "Ggram",
  "fgram",
  "hgram",
  "dagram",
  "mgram",
  "ngram",
  "pgram",
  "ugram",
  "Tgram",
  "ygram",
  "Egram",
  "zgram",
  "Mgram",
  "kgram",
  "Ygram",
  "agramme",
  "cgramme",
  "Zgramme",
  "Pgramme",
  "dgramme",
  "Ggramme",
  "fgramme",
  "hgramme",
  "dagramme",
  "mgramme",
  "ngramme",
  "pgramme",
  "ugramme",
  "Tgramme",
  "ygramme",
  "Egramme",
  "zgramme",
  "Mgramme",
  "kgramme",
  "Ygramme",
  "amolar",
  "cmolar",
  "Zmolar",
  "Pmolar",
  "dmolar",
  "Gmolar",
  "fmolar",
  "hmolar",
  "damolar",
  "mmolar",
  "nmolar",
  "pmolar",
  "umolar",
  "Tmolar",
  "ymolar",
  "Emolar",
  "zmolar",
  "Mmolar",
  "kmolar",
  "Ymolar",
  "aradian",
  "cradian",
  "Zradian",
  "Pradian",
  "dradian",
  "Gradian",
  "fradian",
  "hradian",
  "daradian",
  "mradian",
  "nradian",
  "pradian",
  "uradian",
  "Tradian",
  "yradian",
  "Eradian",
  "zradian",
  "Mradian",
  "kradian",
  "Yradian",
  "asteradian",
  "csteradian",
  "Zsteradian",
  "Psteradian",
  "dsteradian",
  "Gsteradian",
  "fsteradian",
  "hsteradian",
  "dasteradian",
  "msteradian",
  "nsteradian",
  "psteradian",
  "usteradian",
  "Tsteradian",
  "ysteradian",
  "Esteradian",
  "zsteradian",
  "Msteradian",
  "ksteradian",
  "Ysteradian",
  "ahertz",
  "chertz",
  "Zhertz",
  "Phertz",
  "dhertz",
  "Ghertz",
  "fhertz",
  "hhertz",
  "dahertz",
  "mhertz",
  "nhertz",
  "phertz",
  "uhertz",
  "Thertz",
  "yhertz",
  "Ehertz",
  "zhertz",
  "Mhertz",
  "khertz",
  "Yhertz",
  "anewton",
  "cnewton",
  "Znewton",
  "Pnewton",
  "dnewton",
  "Gnewton",
  "fnewton",
  "hnewton",
  "danewton",
  "mnewton",
  "nnewton",
  "pnewton",
  "unewton",
  "Tnewton",
  "ynewton",
  "Enewton",
  "znewton",
  "Mnewton",
  "knewton",
  "Ynewton",
  "apascal",
  "cpascal",
  "Zpascal",
  "Ppascal",
  "dpascal",
  "Gpascal",
  "fpascal",
  "hpascal",
  "dapascal",
  "mpascal",
  "npascal",
  "ppascal",
  "upascal",
  "Tpascal",
  "ypascal",
  "Epascal",
  "zpascal",
  "Mpascal",
  "kpascal",
  "Ypascal",
  "ajoule",
  "cjoule",
  "Zjoule",
  "Pjoule",
  "djoule",
  "Gjoule",
  "fjoule",
  "hjoule",
  "dajoule",
  "mjoule",
  "njoule",
  "pjoule",
  "ujoule",
  "Tjoule",
  "yjoule",
  "Ejoule",
  "zjoule",
  "Mjoule",
  "kjoule",
  "Yjoule",
  "awatt",
  "cwatt",
  "Zwatt",
  "Pwatt",
  "dwatt",
  "Gwatt",
  "fwatt",
  "hwatt",
  "dawatt",
  "mwatt",
  "nwatt",
  "pwatt",
  "uwatt",
  "Twatt",
  "ywatt",
  "Ewatt",
  "zwatt",
  "Mwatt",
  "kwatt",
  "Ywatt",
  "acoulomb",
  "ccoulomb",
  "Zcoulomb",
  "Pcoulomb",
  "dcoulomb",
  "Gcoulomb",
  "fcoulomb",
  "hcoulomb",
  "dacoulomb",
  "mcoulomb",
  "ncoulomb",
  "pcoulomb",
  "ucoulomb",
  "Tcoulomb",
  "ycoulomb",
  "Ecoulomb",
  "zcoulomb",
  "Mcoulomb",
  "kcoulomb",
  "Ycoulomb",
  "avolt",
  "cvolt",
  "Zvolt",
  "Pvolt",
  "dvolt",
  "Gvolt",
  "fvolt",
  "hvolt",
  "davolt",
  "mvolt",
  "nvolt",
  "pvolt",
  "uvolt",
  "Tvolt",
  "yvolt",
  "Evolt",
  "zvolt",
  "Mvolt",
  "kvolt",
  "Yvolt",
  "afarad",
  "cfarad",
  "Zfarad",
  "Pfarad",
  "dfarad",
  "Gfarad",
  "ffarad",
  "hfarad",
  "dafarad",
  "mfarad",
  "nfarad",
  "pfarad",
  "ufarad",
  "Tfarad",
  "yfarad",
  "Efarad",
  "zfarad",
  "Mfarad",
  "kfarad",
  "Yfarad",
  "aohm",
  "cohm",
  "Zohm",
  "Pohm",
  "dohm",
  "Gohm",
  "fohm",
  "hohm",
  "daohm",
  "mohm",
  "nohm",
  "pohm",
  "uohm",
  "Tohm",
  "yohm",
  "Eohm",
  "zohm",
  "Mohm",
  "kohm",
  "Yohm",
  "asiemens",
  "csiemens",
  "Zsiemens",
  "Psiemens",
  "dsiemens",
  "Gsiemens",
  "fsiemens",
  "hsiemens",
  "dasiemens",
  "msiemens",
  "nsiemens",
  "psiemens",
  "usiemens",
  "Tsiemens",
  "ysiemens",
  "Esiemens",
  "zsiemens",
  "Msiemens",
  "ksiemens",
  "Ysiemens",
  "aweber",
  "cweber",
  "Zweber",
  "Pweber",
  "dweber",
  "Gweber",
  "fweber",
  "hweber",
  "daweber",
  "mweber",
  "nweber",
  "pweber",
  "uweber",
  "Tweber",
  "yweber",
  "Eweber",
  "zweber",
  "Mweber",
  "kweber",
  "Yweber",
  "atesla",
  "ctesla",
  "Ztesla",
  "Ptesla",
  "dtesla",
  "Gtesla",
  "ftesla",
  "htesla",
  "datesla",
  "mtesla",
  "ntesla",
  "ptesla",
  "utesla",
  "Ttesla",
  "ytesla",
  "Etesla",
  "ztesla",
  "Mtesla",
  "ktesla",
  "Ytesla",
  "ahenry",
  "chenry",
  "Zhenry",
  "Phenry",
  "dhenry",
  "Ghenry",
  "fhenry",
  "hhenry",
  "dahenry",
  "mhenry",
  "nhenry",
  "phenry",
  "uhenry",
  "Thenry",
  "yhenry",
  "Ehenry",
  "zhenry",
  "Mhenry",
  "khenry",
  "Yhenry",
  "alumen",
  "clumen",
  "Zlumen",
  "Plumen",
  "dlumen",
  "Glumen",
  "flumen",
  "hlumen",
  "dalumen",
  "mlumen",
  "nlumen",
  "plumen",
  "ulumen",
  "Tlumen",
  "ylumen",
  "Elumen",
  "zlumen",
  "Mlumen",
  "klumen",
  "Ylumen",
  "alux",
  "clux",
  "Zlux",
  "Plux",
  "dlux",
  "Glux",
  "flux",
  "hlux",
  "dalux",
  "mlux",
  "nlux",
  "plux",
  "ulux",
  "Tlux",
  "ylux",
  "Elux",
  "zlux",
  "Mlux",
  "klux",
  "Ylux",
  "abecquerel",
  "cbecquerel",
  "Zbecquerel",
  "Pbecquerel",
  "dbecquerel",
  "Gbecquerel",
  "fbecquerel",
  "hbecquerel",
  "dabecquerel",
  "mbecquerel",
  "nbecquerel",
  "pbecquerel",
  "ubecquerel",
  "Tbecquerel",
  "ybecquerel",
  "Ebecquerel",
  "zbecquerel",
  "Mbecquerel",
  "kbecquerel",
  "Ybecquerel",
  "agray",
  "cgray",
  "Zgray",
  "Pgray",
  "dgray",
  "Ggray",
  "fgray",
  "hgray",
  "dagray",
  "mgray",
  "ngray",
  "pgray",
  "ugray",
  "Tgray",
  "ygray",
  "Egray",
  "zgray",
  "Mgray",
  "kgray",
  "Ygray",
  "asievert",
  "csievert",
  "Zsievert",
  "Psievert",
  "dsievert",
  "Gsievert",
  "fsievert",
  "hsievert",
  "dasievert",
  "msievert",
  "nsievert",
  "psievert",
  "usievert",
  "Tsievert",
  "ysievert",
  "Esievert",
  "zsievert",
  "Msievert",
  "ksievert",
  "Ysievert",
  "akatal",
  "ckatal",
  "Zkatal",
  "Pkatal",
  "dkatal",
  "Gkatal",
  "fkatal",
  "hkatal",
  "dakatal",
  "mkatal",
  "nkatal",
  "pkatal",
  "ukatal",
  "Tkatal",
  "ykatal",
  "Ekatal",
  "zkatal",
  "Mkatal",
  "kkatal",
  "Ykatal",
  "metre2",
  "metre3",
  "meter2",
  "meter3",
  "kilogram2",
  "kilogram3",
  "second2",
  "second3",
  "amp2",
  "amp3",
  "ampere2",
  "ampere3",
  "kelvin2",
  "kelvin3",
  "mole2",
  "mole3",
  "mol2",
  "mol3",
  "candle2",
  "candle3",
  "kilogramme2",
  "kilogramme3",
  "gram2",
  "gram3",
  "gramme2",
  "gramme3",
  "molar2",
  "molar3",
  "radian2",
  "radian3",
  "steradian2",
  "steradian3",
  "hertz2",
  "hertz3",
  "newton2",
  "newton3",
  "pascal2",
  "pascal3",
  "joule2",
  "joule3",
  "watt2",
  "watt3",
  "coulomb2",
  "coulomb3",
  "volt2",
  "volt3",
  "farad2",
  "farad3",
  "ohm2",
  "ohm3",
  "siemens2",
  "siemens3",
  "weber2",
  "weber3",
  "tesla2",
  "tesla3",
  "henry2",
  "henry3",
  "lumen2",
  "lumen3",
  "lux2",
  "lux3",
  "becquerel2",
  "becquerel3",
  "gray2",
  "gray3",
  "sievert2",
  "sievert3",
  "katal2",
  "katal3",
  "ametre2",
  "ametre3",
  "cmetre2",
  "cmetre3",
  "Zmetre2",
  "Zmetre3",
  "Pmetre2",
  "Pmetre3",
  "dmetre2",
  "dmetre3",
  "Gmetre2",
  "Gmetre3",
  "fmetre2",
  "fmetre3",
  "hmetre2",
  "hmetre3",
  "dametre2",
  "dametre3",
  "mmetre2",
  "mmetre3",
  "nmetre2",
  "nmetre3",
  "pmetre2",
  "pmetre3",
  "umetre2",
  "umetre3",
  "Tmetre2",
  "Tmetre3",
  "ymetre2",
  "ymetre3",
  "Emetre2",
  "Emetre3",
  "zmetre2",
  "zmetre3",
  "Mmetre2",
  "Mmetre3",
  "kmetre2",
  "kmetre3",
  "Ymetre2",
  "Ymetre3",
  "ameter2",
  "ameter3",
  "cmeter2",
  "cmeter3",
  "Zmeter2",
  "Zmeter3",
  "Pmeter2",
  "Pmeter3",
  "dmeter2",
  "dmeter3",
  "Gmeter2",
  "Gmeter3",
  "fmeter2",
  "fmeter3",
  "hmeter2",
  "hmeter3",
  "dameter2",
  "dameter3",
  "mmeter2",
  "mmeter3",
  "nmeter2",
  "nmeter3",
  "pmeter2",
  "pmeter3",
  "umeter2",
  "umeter3",
  "Tmeter2",
  "Tmeter3",
  "ymeter2",
  "ymeter3",
  "Emeter2",
  "Emeter3",
  "zmeter2",
  "zmeter3",
  "Mmeter2",
  "Mmeter3",
  "kmeter2",
  "kmeter3",
  "Ymeter2",
  "Ymeter3",
  "asecond2",
  "asecond3",
  "csecond2",
  "csecond3",
  "Zsecond2",
  "Zsecond3",
  "Psecond2",
  "Psecond3",
  "dsecond2",
  "dsecond3",
  "Gsecond2",
  "Gsecond3",
  "fsecond2",
  "fsecond3",
  "hsecond2",
  "hsecond3",
  "dasecond2",
  "dasecond3",
  "msecond2",
  "msecond3",
  "nsecond2",
  "nsecond3",
  "psecond2",
  "psecond3",
  "usecond2",
  "usecond3",
  "Tsecond2",
  "Tsecond3",
  "ysecond2",
  "ysecond3",
  "Esecond2",
  "Esecond3",
  "zsecond2",
  "zsecond3",
  "Msecond2",
  "Msecond3",
  "ksecond2",
  "ksecond3",
  "Ysecond2",
  "Ysecond3",
  "aamp2",
  "aamp3",
  "camp2",
  "camp3",
  "Zamp2",
  "Zamp3",
  "Pamp2",
  "Pamp3",
  "damp2",
  "damp3",
  "Gamp2",
  "Gamp3",
  "famp2",
  "famp3",
  "hamp2",
  "hamp3",
  "daamp2",
  "daamp3",
  "mamp2",
  "mamp3",
  "namp2",
  "namp3",
  "pamp2",
  "pamp3",
  "uamp2",
  "uamp3",
  "Tamp2",
  "Tamp3",
  "yamp2",
  "yamp3",
  "Eamp2",
  "Eamp3",
  "zamp2",
  "zamp3",
  "Mamp2",
  "Mamp3",
  "kamp2",
  "kamp3",
  "Yamp2",
  "Yamp3",
  "aampere2",
  "aampere3",
  "campere2",
  "campere3",
  "Zampere2",
  "Zampere3",
  "Pampere2",
  "Pampere3",
  "dampere2",
  "dampere3",
  "Gampere2",
  "Gampere3",
  "fampere2",
  "fampere3",
  "hampere2",
  "hampere3",
  "daampere2",
  "daampere3",
  "mampere2",
  "mampere3",
  "nampere2",
  "nampere3",
  "pampere2",
  "pampere3",
  "uampere2",
  "uampere3",
  "Tampere2",
  "Tampere3",
  "yampere2",
  "yampere3",
  "Eampere2",
  "Eampere3",
  "zampere2",
  "zampere3",
  "Mampere2",
  "Mampere3",
  "kampere2",
  "kampere3",
  "Yampere2",
  "Yampere3",
  "amole2",
  "amole3",
  "cmole2",
  "cmole3",
  "Zmole2",
  "Zmole3",
  "Pmole2",
  "Pmole3",
  "dmole2",
  "dmole3",
  "Gmole2",
  "Gmole3",
  "fmole2",
  "fmole3",
  "hmole2",
  "hmole3",
  "damole2",
  "damole3",
  "mmole2",
  "mmole3",
  "nmole2",
  "nmole3",
  "pmole2",
  "pmole3",
  "umole2",
  "umole3",
  "Tmole2",
  "Tmole3",
  "ymole2",
  "ymole3",
  "Emole2",
  "Emole3",
  "zmole2",
  "zmole3",
  "Mmole2",
  "Mmole3",
  "kmole2",
  "kmole3",
  "Ymole2",
  "Ymole3",
  "amol2",
  "amol3",
  "cmol2",
  "cmol3",
  "Zmol2",
  "Zmol3",
  "Pmol2",
  "Pmol3",
  "dmol2",
  "dmol3",
  "Gmol2",
  "Gmol3",
  "fmol2",
  "fmol3",
  "hmol2",
  "hmol3",
  "damol2",
  "damol3",
  "mmol2",
  "mmol3",
  "nmol2",
  "nmol3",
  "pmol2",
  "pmol3",
  "umol2",
  "umol3",
  "Tmol2",
  "Tmol3",
  "ymol2",
  "ymol3",
  "Emol2",
  "Emol3",
  "zmol2",
  "zmol3",
  "Mmol2",
  "Mmol3",
  "kmol2",
  "kmol3",
  "Ymol2",
  "Ymol3",
  "acandle2",
  "acandle3",
  "ccandle2",
  "ccandle3",
  "Zcandle2",
  "Zcandle3",
  "Pcandle2",
  "Pcandle3",
  "dcandle2",
  "dcandle3",
  "Gcandle2",
  "Gcandle3",
  "fcandle2",
  "fcandle3",
  "hcandle2",
  "hcandle3",
  "dacandle2",
  "dacandle3",
  "mcandle2",
  "mcandle3",
  "ncandle2",
  "ncandle3",
  "pcandle2",
  "pcandle3",
  "ucandle2",
  "ucandle3",
  "Tcandle2",
  "Tcandle3",
  "ycandle2",
  "ycandle3",
  "Ecandle2",
  "Ecandle3",
  "zcandle2",
  "zcandle3",
  "Mcandle2",
  "Mcandle3",
  "kcandle2",
  "kcandle3",
  "Ycandle2",
  "Ycandle3",
  "agram2",
  "agram3",
  "cgram2",
  "cgram3",
  "Zgram2",
  "Zgram3",
  "Pgram2",
  "Pgram3",
  "dgram2",
  "dgram3",
  "Ggram2",
  "Ggram3",
  "fgram2",
  "fgram3",
  "hgram2",
  "hgram3",
  "dagram2",
  "dagram3",
  "mgram2",
  "mgram3",
  "ngram2",
  "ngram3",
  "pgram2",
  "pgram3",
  "ugram2",
  "ugram3",
  "Tgram2",
  "Tgram3",
  "ygram2",
  "ygram3",
  "Egram2",
  "Egram3",
  "zgram2",
  "zgram3",
  "Mgram2",
  "Mgram3",
  "kgram2",
  "kgram3",
  "Ygram2",
  "Ygram3",
  "agramme2",
  "agramme3",
  "cgramme2",
  "cgramme3",
  "Zgramme2",
  "Zgramme3",
  "Pgramme2",
  "Pgramme3",
  "dgramme2",
  "dgramme3",
  "Ggramme2",
  "Ggramme3",
  "fgramme2",
  "fgramme3",
  "hgramme2",
  "hgramme3",
  "dagramme2",
  "dagramme3",
  "mgramme2",
  "mgramme3",
  "ngramme2",
  "ngramme3",
  "pgramme2",
  "pgramme3",
  "ugramme2",
  "ugramme3",
  "Tgramme2",
  "Tgramme3",
  "ygramme2",
  "ygramme3",
  "Egramme2",
  "Egramme3",
  "zgramme2",
  "zgramme3",
  "Mgramme2",
  "Mgramme3",
  "kgramme2",
  "kgramme3",
  "Ygramme2",
  "Ygramme3",
  "amolar2",
  "amolar3",
  "cmolar2",
  "cmolar3",
  "Zmolar2",
  "Zmolar3",
  "Pmolar2",
  "Pmolar3",
  "dmolar2",
  "dmolar3",
  "Gmolar2",
  "Gmolar3",
  "fmolar2",
  "fmolar3",
  "hmolar2",
  "hmolar3",
  "damolar2",
  "damolar3",
  "mmolar2",
  "mmolar3",
  "nmolar2",
  "nmolar3",
  "pmolar2",
  "pmolar3",
  "umolar2",
  "umolar3",
  "Tmolar2",
  "Tmolar3",
  "ymolar2",
  "ymolar3",
  "Emolar2",
  "Emolar3",
  "zmolar2",
  "zmolar3",
  "Mmolar2",
  "Mmolar3",
  "kmolar2",
  "kmolar3",
  "Ymolar2",
  "Ymolar3",
  "aradian2",
  "aradian3",
  "cradian2",
  "cradian3",
  "Zradian2",
  "Zradian3",
  "Pradian2",
  "Pradian3",
  "dradian2",
  "dradian3",
  "Gradian2",
  "Gradian3",
  "fradian2",
  "fradian3",
  "hradian2",
  "hradian3",
  "daradian2",
  "daradian3",
  "mradian2",
  "mradian3",
  "nradian2",
  "nradian3",
  "pradian2",
  "pradian3",
  "uradian2",
  "uradian3",
  "Tradian2",
  "Tradian3",
  "yradian2",
  "yradian3",
  "Eradian2",
  "Eradian3",
  "zradian2",
  "zradian3",
  "Mradian2",
  "Mradian3",
  "kradian2",
  "kradian3",
  "Yradian2",
  "Yradian3",
  "asteradian2",
  "asteradian3",
  "csteradian2",
  "csteradian3",
  "Zsteradian2",
  "Zsteradian3",
  "Psteradian2",
  "Psteradian3",
  "dsteradian2",
  "dsteradian3",
  "Gsteradian2",
  "Gsteradian3",
  "fsteradian2",
  "fsteradian3",
  "hsteradian2",
  "hsteradian3",
  "dasteradian2",
  "dasteradian3",
  "msteradian2",
  "msteradian3",
  "nsteradian2",
  "nsteradian3",
  "psteradian2",
  "psteradian3",
  "usteradian2",
  "usteradian3",
  "Tsteradian2",
  "Tsteradian3",
  "ysteradian2",
  "ysteradian3",
  "Esteradian2",
  "Esteradian3",
  "zsteradian2",
  "zsteradian3",
  "Msteradian2",
  "Msteradian3",
  "ksteradian2",
  "ksteradian3",
  "Ysteradian2",
  "Ysteradian3",
  "ahertz2",
  "ahertz3",
  "chertz2",
  "chertz3",
  "Zhertz2",
  "Zhertz3",
  "Phertz2",
  "Phertz3",
  "dhertz2",
  "dhertz3",
  "Ghertz2",
  "Ghertz3",
  "fhertz2",
  "fhertz3",
  "hhertz2",
  "hhertz3",
  "dahertz2",
  "dahertz3",
  "mhertz2",
  "mhertz3",
  "nhertz2",
  "nhertz3",
  "phertz2",
  "phertz3",
  "uhertz2",
  "uhertz3",
  "Thertz2",
  "Thertz3",
  "yhertz2",
  "yhertz3",
  "Ehertz2",
  "Ehertz3",
  "zhertz2",
  "zhertz3",
  "Mhertz2",
  "Mhertz3",
  "khertz2",
  "khertz3",
  "Yhertz2",
  "Yhertz3",
  "anewton2",
  "anewton3",
  "cnewton2",
  "cnewton3",
  "Znewton2",
  "Znewton3",
  "Pnewton2",
  "Pnewton3",
  "dnewton2",
  "dnewton3",
  "Gnewton2",
  "Gnewton3",
  "fnewton2",
  "fnewton3",
  "hnewton2",
  "hnewton3",
  "danewton2",
  "danewton3",
  "mnewton2",
  "mnewton3",
  "nnewton2",
  "nnewton3",
  "pnewton2",
  "pnewton3",
  "unewton2",
  "unewton3",
  "Tnewton2",
  "Tnewton3",
  "ynewton2",
  "ynewton3",
  "Enewton2",
  "Enewton3",
  "znewton2",
  "znewton3",
  "Mnewton2",
  "Mnewton3",
  "knewton2",
  "knewton3",
  "Ynewton2",
  "Ynewton3",
  "apascal2",
  "apascal3",
  "cpascal2",
  "cpascal3",
  "Zpascal2",
  "Zpascal3",
  "Ppascal2",
  "Ppascal3",
  "dpascal2",
  "dpascal3",
  "Gpascal2",
  "Gpascal3",
  "fpascal2",
  "fpascal3",
  "hpascal2",
  "hpascal3",
  "dapascal2",
  "dapascal3",
  "mpascal2",
  "mpascal3",
  "npascal2",
  "npascal3",
  "ppascal2",
  "ppascal3",
  "upascal2",
  "upascal3",
  "Tpascal2",
  "Tpascal3",
  "ypascal2",
  "ypascal3",
  "Epascal2",
  "Epascal3",
  "zpascal2",
  "zpascal3",
  "Mpascal2",
  "Mpascal3",
  "kpascal2",
  "kpascal3",
  "Ypascal2",
  "Ypascal3",
  "ajoule2",
  "ajoule3",
  "cjoule2",
  "cjoule3",
  "Zjoule2",
  "Zjoule3",
  "Pjoule2",
  "Pjoule3",
  "djoule2",
  "djoule3",
  "Gjoule2",
  "Gjoule3",
  "fjoule2",
  "fjoule3",
  "hjoule2",
  "hjoule3",
  "dajoule2",
  "dajoule3",
  "mjoule2",
  "mjoule3",
  "njoule2",
  "njoule3",
  "pjoule2",
  "pjoule3",
  "ujoule2",
  "ujoule3",
  "Tjoule2",
  "Tjoule3",
  "yjoule2",
  "yjoule3",
  "Ejoule2",
  "Ejoule3",
  "zjoule2",
  "zjoule3",
  "Mjoule2",
  "Mjoule3",
  "kjoule2",
  "kjoule3",
  "Yjoule2",
  "Yjoule3",
  "awatt2",
  "awatt3",
  "cwatt2",
  "cwatt3",
  "Zwatt2",
  "Zwatt3",
  "Pwatt2",
  "Pwatt3",
  "dwatt2",
  "dwatt3",
  "Gwatt2",
  "Gwatt3",
  "fwatt2",
  "fwatt3",
  "hwatt2",
  "hwatt3",
  "dawatt2",
  "dawatt3",
  "mwatt2",
  "mwatt3",
  "nwatt2",
  "nwatt3",
  "pwatt2",
  "pwatt3",
  "uwatt2",
  "uwatt3",
  "Twatt2",
  "Twatt3",
  "ywatt2",
  "ywatt3",
  "Ewatt2",
  "Ewatt3",
  "zwatt2",
  "zwatt3",
  "Mwatt2",
  "Mwatt3",
  "kwatt2",
  "kwatt3",
  "Ywatt2",
  "Ywatt3",
  "acoulomb2",
  "acoulomb3",
  "ccoulomb2",
  "ccoulomb3",
  "Zcoulomb2",
  "Zcoulomb3",
  "Pcoulomb2",
  "Pcoulomb3",
  "dcoulomb2",
  "dcoulomb3",
  "Gcoulomb2",
  "Gcoulomb3",
  "fcoulomb2",
  "fcoulomb3",
  "hcoulomb2",
  "hcoulomb3",
  "dacoulomb2",
  "dacoulomb3",
  "mcoulomb2",
  "mcoulomb3",
  "ncoulomb2",
  "ncoulomb3",
  "pcoulomb2",
  "pcoulomb3",
  "ucoulomb2",
  "ucoulomb3",
  "Tcoulomb2",
  "Tcoulomb3",
  "ycoulomb2",
  "ycoulomb3",
  "Ecoulomb2",
  "Ecoulomb3",
  "zcoulomb2",
  "zcoulomb3",
  "Mcoulomb2",
  "Mcoulomb3",
  "kcoulomb2",
  "kcoulomb3",
  "Ycoulomb2",
  "Ycoulomb3",
  "avolt2",
  "avolt3",
  "cvolt2",
  "cvolt3",
  "Zvolt2",
  "Zvolt3",
  "Pvolt2",
  "Pvolt3",
  "dvolt2",
  "dvolt3",
  "Gvolt2",
  "Gvolt3",
  "fvolt2",
  "fvolt3",
  "hvolt2",
  "hvolt3",
  "davolt2",
  "davolt3",
  "mvolt2",
  "mvolt3",
  "nvolt2",
  "nvolt3",
  "pvolt2",
  "pvolt3",
  "uvolt2",
  "uvolt3",
  "Tvolt2",
  "Tvolt3",
  "yvolt2",
  "yvolt3",
  "Evolt2",
  "Evolt3",
  "zvolt2",
  "zvolt3",
  "Mvolt2",
  "Mvolt3",
  "kvolt2",
  "kvolt3",
  "Yvolt2",
  "Yvolt3",
  "afarad2",
  "afarad3",
  "cfarad2",
  "cfarad3",
  "Zfarad2",
  "Zfarad3",
  "Pfarad2",
  "Pfarad3",
  "dfarad2",
  "dfarad3",
  "Gfarad2",
  "Gfarad3",
  "ffarad2",
  "ffarad3",
  "hfarad2",
  "hfarad3",
  "dafarad2",
  "dafarad3",
  "mfarad2",
  "mfarad3",
  "nfarad2",
  "nfarad3",
  "pfarad2",
  "pfarad3",
  "ufarad2",
  "ufarad3",
  "Tfarad2",
  "Tfarad3",
  "yfarad2",
  "yfarad3",
  "Efarad2",
  "Efarad3",
  "zfarad2",
  "zfarad3",
  "Mfarad2",
  "Mfarad3",
  "kfarad2",
  "kfarad3",
  "Yfarad2",
  "Yfarad3",
  "aohm2",
  "aohm3",
  "cohm2",
  "cohm3",
  "Zohm2",
  "Zohm3",
  "Pohm2",
  "Pohm3",
  "dohm2",
  "dohm3",
  "Gohm2",
  "Gohm3",
  "fohm2",
  "fohm3",
  "hohm2",
  "hohm3",
  "daohm2",
  "daohm3",
  "mohm2",
  "mohm3",
  "nohm2",
  "nohm3",
  "pohm2",
  "pohm3",
  "uohm2",
  "uohm3",
  "Tohm2",
  "Tohm3",
  "yohm2",
  "yohm3",
  "Eohm2",
  "Eohm3",
  "zohm2",
  "zohm3",
  "Mohm2",
  "Mohm3",
  "kohm2",
  "kohm3",
  "Yohm2",
  "Yohm3",
  "asiemens2",
  "asiemens3",
  "csiemens2",
  "csiemens3",
  "Zsiemens2",
  "Zsiemens3",
  "Psiemens2",
  "Psiemens3",
  "dsiemens2",
  "dsiemens3",
  "Gsiemens2",
  "Gsiemens3",
  "fsiemens2",
  "fsiemens3",
  "hsiemens2",
  "hsiemens3",
  "dasiemens2",
  "dasiemens3",
  "msiemens2",
  "msiemens3",
  "nsiemens2",
  "nsiemens3",
  "psiemens2",
  "psiemens3",
  "usiemens2",
  "usiemens3",
  "Tsiemens2",
  "Tsiemens3",
  "ysiemens2",
  "ysiemens3",
  "Esiemens2",
  "Esiemens3",
  "zsiemens2",
  "zsiemens3",
  "Msiemens2",
  "Msiemens3",
  "ksiemens2",
  "ksiemens3",
  "Ysiemens2",
  "Ysiemens3",
  "aweber2",
  "aweber3",
  "cweber2",
  "cweber3",
  "Zweber2",
  "Zweber3",
  "Pweber2",
  "Pweber3",
  "dweber2",
  "dweber3",
  "Gweber2",
  "Gweber3",
  "fweber2",
  "fweber3",
  "hweber2",
  "hweber3",
  "daweber2",
  "daweber3",
  "mweber2",
  "mweber3",
  "nweber2",
  "nweber3",
  "pweber2",
  "pweber3",
  "uweber2",
  "uweber3",
  "Tweber2",
  "Tweber3",
  "yweber2",
  "yweber3",
  "Eweber2",
  "Eweber3",
  "zweber2",
  "zweber3",
  "Mweber2",
  "Mweber3",
  "kweber2",
  "kweber3",
  "Yweber2",
  "Yweber3",
  "atesla2",
  "atesla3",
  "ctesla2",
  "ctesla3",
  "Ztesla2",
  "Ztesla3",
  "Ptesla2",
  "Ptesla3",
  "dtesla2",
  "dtesla3",
  "Gtesla2",
  "Gtesla3",
  "ftesla2",
  "ftesla3",
  "htesla2",
  "htesla3",
  "datesla2",
  "datesla3",
  "mtesla2",
  "mtesla3",
  "ntesla2",
  "ntesla3",
  "ptesla2",
  "ptesla3",
  "utesla2",
  "utesla3",
  "Ttesla2",
  "Ttesla3",
  "ytesla2",
  "ytesla3",
  "Etesla2",
  "Etesla3",
  "ztesla2",
  "ztesla3",
  "Mtesla2",
  "Mtesla3",
  "ktesla2",
  "ktesla3",
  "Ytesla2",
  "Ytesla3",
  "ahenry2",
  "ahenry3",
  "chenry2",
  "chenry3",
  "Zhenry2",
  "Zhenry3",
  "Phenry2",
  "Phenry3",
  "dhenry2",
  "dhenry3",
  "Ghenry2",
  "Ghenry3",
  "fhenry2",
  "fhenry3",
  "hhenry2",
  "hhenry3",
  "dahenry2",
  "dahenry3",
  "mhenry2",
  "mhenry3",
  "nhenry2",
  "nhenry3",
  "phenry2",
  "phenry3",
  "uhenry2",
  "uhenry3",
  "Thenry2",
  "Thenry3",
  "yhenry2",
  "yhenry3",
  "Ehenry2",
  "Ehenry3",
  "zhenry2",
  "zhenry3",
  "Mhenry2",
  "Mhenry3",
  "khenry2",
  "khenry3",
  "Yhenry2",
  "Yhenry3",
  "alumen2",
  "alumen3",
  "clumen2",
  "clumen3",
  "Zlumen2",
  "Zlumen3",
  "Plumen2",
  "Plumen3",
  "dlumen2",
  "dlumen3",
  "Glumen2",
  "Glumen3",
  "flumen2",
  "flumen3",
  "hlumen2",
  "hlumen3",
  "dalumen2",
  "dalumen3",
  "mlumen2",
  "mlumen3",
  "nlumen2",
  "nlumen3",
  "plumen2",
  "plumen3",
  "ulumen2",
  "ulumen3",
  "Tlumen2",
  "Tlumen3",
  "ylumen2",
  "ylumen3",
  "Elumen2",
  "Elumen3",
  "zlumen2",
  "zlumen3",
  "Mlumen2",
  "Mlumen3",
  "klumen2",
  "klumen3",
  "Ylumen2",
  "Ylumen3",
  "alux2",
  "alux3",
  "clux2",
  "clux3",
  "Zlux2",
  "Zlux3",
  "Plux2",
  "Plux3",
  "dlux2",
  "dlux3",
  "Glux2",
  "Glux3",
  "flux2",
  "flux3",
  "hlux2",
  "hlux3",
  "dalux2",
  "dalux3",
  "mlux2",
  "mlux3",
  "nlux2",
  "nlux3",
  "plux2",
  "plux3",
  "ulux2",
  "ulux3",
  "Tlux2",
  "Tlux3",
  "ylux2",
  "ylux3",
  "Elux2",
  "Elux3",
  "zlux2",
  "zlux3",
  "Mlux2",
  "Mlux3",
  "klux2",
  "klux3",
  "Ylux2",
  "Ylux3",
  "abecquerel2",
  "abecquerel3",
  "cbecquerel2",
  "cbecquerel3",
  "Zbecquerel2",
  "Zbecquerel3",
  "Pbecquerel2",
  "Pbecquerel3",
  "dbecquerel2",
  "dbecquerel3",
  "Gbecquerel2",
  "Gbecquerel3",
  "fbecquerel2",
  "fbecquerel3",
  "hbecquerel2",
  "hbecquerel3",
  "dabecquerel2",
  "dabecquerel3",
  "mbecquerel2",
  "mbecquerel3",
  "nbecquerel2",
  "nbecquerel3",
  "pbecquerel2",
  "pbecquerel3",
  "ubecquerel2",
  "ubecquerel3",
  "Tbecquerel2",
  "Tbecquerel3",
  "ybecquerel2",
  "ybecquerel3",
  "Ebecquerel2",
  "Ebecquerel3",
  "zbecquerel2",
  "zbecquerel3",
  "Mbecquerel2",
  "Mbecquerel3",
  "kbecquerel2",
  "kbecquerel3",
  "Ybecquerel2",
  "Ybecquerel3",
  "agray2",
  "agray3",
  "cgray2",
  "cgray3",
  "Zgray2",
  "Zgray3",
  "Pgray2",
  "Pgray3",
  "dgray2",
  "dgray3",
  "Ggray2",
  "Ggray3",
  "fgray2",
  "fgray3",
  "hgray2",
  "hgray3",
  "dagray2",
  "dagray3",
  "mgray2",
  "mgray3",
  "ngray2",
  "ngray3",
  "pgray2",
  "pgray3",
  "ugray2",
  "ugray3",
  "Tgray2",
  "Tgray3",
  "ygray2",
  "ygray3",
  "Egray2",
  "Egray3",
  "zgray2",
  "zgray3",
  "Mgray2",
  "Mgray3",
  "kgray2",
  "kgray3",
  "Ygray2",
  "Ygray3",
  "asievert2",
  "asievert3",
  "csievert2",
  "csievert3",
  "Zsievert2",
  "Zsievert3",
  "Psievert2",
  "Psievert3",
  "dsievert2",
  "dsievert3",
  "Gsievert2",
  "Gsievert3",
  "fsievert2",
  "fsievert3",
  "hsievert2",
  "hsievert3",
  "dasievert2",
  "dasievert3",
  "msievert2",
  "msievert3",
  "nsievert2",
  "nsievert3",
  "psievert2",
  "psievert3",
  "usievert2",
  "usievert3",
  "Tsievert2",
  "Tsievert3",
  "ysievert2",
  "ysievert3",
  "Esievert2",
  "Esievert3",
  "zsievert2",
  "zsievert3",
  "Msievert2",
  "Msievert3",
  "ksievert2",
  "ksievert3",
  "Ysievert2",
  "Ysievert3",
  "akatal2",
  "akatal3",
  "ckatal2",
  "ckatal3",
  "Zkatal2",
  "Zkatal3",
  "Pkatal2",
  "Pkatal3",
  "dkatal2",
  "dkatal3",
  "Gkatal2",
  "Gkatal3",
  "fkatal2",
  "fkatal3",
  "hkatal2",
  "hkatal3",
  "dakatal2",
  "dakatal3",
  "mkatal2",
  "mkatal3",
  "nkatal2",
  "nkatal3",
  "pkatal2",
  "pkatal3",
  "ukatal2",
  "ukatal3",
  "Tkatal2",
  "Tkatal3",
  "ykatal2",
  "ykatal3",
  "Ekatal2",
  "Ekatal3",
  "zkatal2",
  "zkatal3",
  "Mkatal2",
  "Mkatal3",
  "kkatal2",
  "kkatal3",
  "Ykatal2",
  "Ykatal3",
  "liter",
  "aliter",
  "liter",
  "cliter",
  "Zliter",
  "Pliter",
  "dliter",
  "Gliter",
  "fliter",
  "hliter",
  "daliter",
  "mliter",
  "nliter",
  "pliter",
  "uliter",
  "Tliter",
  "yliter",
  "Eliter",
  "zliter",
  "Mliter",
  "kliter",
  "Yliter",
  "litre",
  "alitre",
  "litre",
  "clitre",
  "Zlitre",
  "Plitre",
  "dlitre",
  "Glitre",
  "flitre",
  "hlitre",
  "dalitre",
  "mlitre",
  "nlitre",
  "plitre",
  "ulitre",
  "Tlitre",
  "ylitre",
  "Elitre",
  "zlitre",
  "Mlitre",
  "klitre",
  "Ylitre",
]

#### FUNDAMENTAL UNITS
metre = Unit.create(get_or_create_dimension(m=1), "metre", "m")
meter = Unit.create(get_or_create_dimension(m=1), "meter", "m")
# Liter has a scale of 10^-3, since 1 l = 1 dm^3 = 10^-3 m^3
liter = Unit.create((meter ** 3).dim, name="liter", dispname="l", scale=-3)
litre = Unit.create((meter ** 3).dim, name="litre", dispname="l", scale=-3)
kilogram = Unit.create(get_or_create_dimension(kg=1), "kilogram", "kg")
kilogramme = Unit.create(get_or_create_dimension(kg=1), "kilogramme", "kg")
gram = Unit.create(kilogram.dim, name="gram", dispname="g", scale=-3)
gramme = Unit.create(kilogram.dim, name="gramme", dispname="g", scale=-3)
second = Unit.create(get_or_create_dimension(s=1), "second", "s")
amp = Unit.create(get_or_create_dimension(A=1), "amp", "A")
ampere = Unit.create(get_or_create_dimension(A=1), "ampere", "A")
kelvin = Unit.create(get_or_create_dimension(K=1), "kelvin", "K")
mole = Unit.create(get_or_create_dimension(mol=1), "mole", "mol")
mol = Unit.create(get_or_create_dimension(mol=1), "mol", "mol")
# Molar has a scale of 10^3, since 1 M = 1 mol/l = 1000 mol/m^3
molar = Unit.create((mole / liter).dim, name="molar", dispname="M", scale=3)
candle = Unit.create(get_or_create_dimension(candle=1), "candle", "cd")
fundamental_units = [metre, meter, gram, second, amp, kelvin, mole, candle]

radian = Unit.create(get_or_create_dimension(), "radian", "rad")
steradian = Unit.create(get_or_create_dimension(), "steradian", "sr")
hertz = Unit.create(get_or_create_dimension(s=-1), "hertz", "Hz")
newton = Unit.create(get_or_create_dimension(m=1, kg=1, s=-2), "newton", "N")
pascal = Unit.create(get_or_create_dimension(m=-1, kg=1, s=-2), "pascal", "Pa")
joule = Unit.create(get_or_create_dimension(m=2, kg=1, s=-2), "joule", "J")
watt = Unit.create(get_or_create_dimension(m=2, kg=1, s=-3), "watt", "W")
coulomb = Unit.create(get_or_create_dimension(s=1, A=1), "coulomb", "C")
volt = Unit.create(get_or_create_dimension(m=2, kg=1, s=-3, A=-1), "volt", "V")
farad = Unit.create(get_or_create_dimension(m=-2, kg=-1, s=4, A=2), "farad", "F")
ohm = Unit.create(get_or_create_dimension(m=2, kg=1, s=-3, A=-2), "ohm", "ohm")
siemens = Unit.create(get_or_create_dimension(m=-2, kg=-1, s=3, A=2), "siemens", "S")
weber = Unit.create(get_or_create_dimension(m=2, kg=1, s=-2, A=-1), "weber", "Wb")
tesla = Unit.create(get_or_create_dimension(kg=1, s=-2, A=-1), "tesla", "T")
henry = Unit.create(get_or_create_dimension(m=2, kg=1, s=-2, A=-2), "henry", "H")
lumen = Unit.create(get_or_create_dimension(cd=1), "lumen", "lm")
lux = Unit.create(get_or_create_dimension(m=-2, cd=1), "lux", "lx")
becquerel = Unit.create(get_or_create_dimension(s=-1), "becquerel", "Bq")
gray = Unit.create(get_or_create_dimension(m=2, s=-2), "gray", "Gy")
sievert = Unit.create(get_or_create_dimension(m=2, s=-2), "sievert", "Sv")
katal = Unit.create(get_or_create_dimension(s=-1, mol=1), "katal", "kat")

######### SCALED BASE UNITS ###########
ametre = Unit.create_scaled_unit(metre, "a")
cmetre = Unit.create_scaled_unit(metre, "c")
Zmetre = Unit.create_scaled_unit(metre, "Z")
Pmetre = Unit.create_scaled_unit(metre, "P")
dmetre = Unit.create_scaled_unit(metre, "d")
Gmetre = Unit.create_scaled_unit(metre, "G")
fmetre = Unit.create_scaled_unit(metre, "f")
hmetre = Unit.create_scaled_unit(metre, "h")
dametre = Unit.create_scaled_unit(metre, "da")
mmetre = Unit.create_scaled_unit(metre, "m")
nmetre = Unit.create_scaled_unit(metre, "n")
pmetre = Unit.create_scaled_unit(metre, "p")
umetre = Unit.create_scaled_unit(metre, "u")
Tmetre = Unit.create_scaled_unit(metre, "T")
ymetre = Unit.create_scaled_unit(metre, "y")
Emetre = Unit.create_scaled_unit(metre, "E")
zmetre = Unit.create_scaled_unit(metre, "z")
Mmetre = Unit.create_scaled_unit(metre, "M")
kmetre = Unit.create_scaled_unit(metre, "k")
Ymetre = Unit.create_scaled_unit(metre, "Y")
ameter = Unit.create_scaled_unit(meter, "a")
cmeter = Unit.create_scaled_unit(meter, "c")
Zmeter = Unit.create_scaled_unit(meter, "Z")
Pmeter = Unit.create_scaled_unit(meter, "P")
dmeter = Unit.create_scaled_unit(meter, "d")
Gmeter = Unit.create_scaled_unit(meter, "G")
fmeter = Unit.create_scaled_unit(meter, "f")
hmeter = Unit.create_scaled_unit(meter, "h")
dameter = Unit.create_scaled_unit(meter, "da")
mmeter = Unit.create_scaled_unit(meter, "m")
nmeter = Unit.create_scaled_unit(meter, "n")
pmeter = Unit.create_scaled_unit(meter, "p")
umeter = Unit.create_scaled_unit(meter, "u")
Tmeter = Unit.create_scaled_unit(meter, "T")
ymeter = Unit.create_scaled_unit(meter, "y")
Emeter = Unit.create_scaled_unit(meter, "E")
zmeter = Unit.create_scaled_unit(meter, "z")
Mmeter = Unit.create_scaled_unit(meter, "M")
kmeter = Unit.create_scaled_unit(meter, "k")
Ymeter = Unit.create_scaled_unit(meter, "Y")
asecond = Unit.create_scaled_unit(second, "a")
csecond = Unit.create_scaled_unit(second, "c")
Zsecond = Unit.create_scaled_unit(second, "Z")
Psecond = Unit.create_scaled_unit(second, "P")
dsecond = Unit.create_scaled_unit(second, "d")
Gsecond = Unit.create_scaled_unit(second, "G")
fsecond = Unit.create_scaled_unit(second, "f")
hsecond = Unit.create_scaled_unit(second, "h")
dasecond = Unit.create_scaled_unit(second, "da")
msecond = Unit.create_scaled_unit(second, "m")
nsecond = Unit.create_scaled_unit(second, "n")
psecond = Unit.create_scaled_unit(second, "p")
usecond = Unit.create_scaled_unit(second, "u")
Tsecond = Unit.create_scaled_unit(second, "T")
ysecond = Unit.create_scaled_unit(second, "y")
Esecond = Unit.create_scaled_unit(second, "E")
zsecond = Unit.create_scaled_unit(second, "z")
Msecond = Unit.create_scaled_unit(second, "M")
ksecond = Unit.create_scaled_unit(second, "k")
Ysecond = Unit.create_scaled_unit(second, "Y")
aamp = Unit.create_scaled_unit(amp, "a")
camp = Unit.create_scaled_unit(amp, "c")
Zamp = Unit.create_scaled_unit(amp, "Z")
Pamp = Unit.create_scaled_unit(amp, "P")
damp = Unit.create_scaled_unit(amp, "d")
Gamp = Unit.create_scaled_unit(amp, "G")
famp = Unit.create_scaled_unit(amp, "f")
hamp = Unit.create_scaled_unit(amp, "h")
daamp = Unit.create_scaled_unit(amp, "da")
mamp = Unit.create_scaled_unit(amp, "m")
namp = Unit.create_scaled_unit(amp, "n")
pamp = Unit.create_scaled_unit(amp, "p")
uamp = Unit.create_scaled_unit(amp, "u")
Tamp = Unit.create_scaled_unit(amp, "T")
yamp = Unit.create_scaled_unit(amp, "y")
Eamp = Unit.create_scaled_unit(amp, "E")
zamp = Unit.create_scaled_unit(amp, "z")
Mamp = Unit.create_scaled_unit(amp, "M")
kamp = Unit.create_scaled_unit(amp, "k")
Yamp = Unit.create_scaled_unit(amp, "Y")
aampere = Unit.create_scaled_unit(ampere, "a")
campere = Unit.create_scaled_unit(ampere, "c")
Zampere = Unit.create_scaled_unit(ampere, "Z")
Pampere = Unit.create_scaled_unit(ampere, "P")
dampere = Unit.create_scaled_unit(ampere, "d")
Gampere = Unit.create_scaled_unit(ampere, "G")
fampere = Unit.create_scaled_unit(ampere, "f")
hampere = Unit.create_scaled_unit(ampere, "h")
daampere = Unit.create_scaled_unit(ampere, "da")
mampere = Unit.create_scaled_unit(ampere, "m")
nampere = Unit.create_scaled_unit(ampere, "n")
pampere = Unit.create_scaled_unit(ampere, "p")
uampere = Unit.create_scaled_unit(ampere, "u")
Tampere = Unit.create_scaled_unit(ampere, "T")
yampere = Unit.create_scaled_unit(ampere, "y")
Eampere = Unit.create_scaled_unit(ampere, "E")
zampere = Unit.create_scaled_unit(ampere, "z")
Mampere = Unit.create_scaled_unit(ampere, "M")
kampere = Unit.create_scaled_unit(ampere, "k")
Yampere = Unit.create_scaled_unit(ampere, "Y")
amole = Unit.create_scaled_unit(mole, "a")
cmole = Unit.create_scaled_unit(mole, "c")
Zmole = Unit.create_scaled_unit(mole, "Z")
Pmole = Unit.create_scaled_unit(mole, "P")
dmole = Unit.create_scaled_unit(mole, "d")
Gmole = Unit.create_scaled_unit(mole, "G")
fmole = Unit.create_scaled_unit(mole, "f")
hmole = Unit.create_scaled_unit(mole, "h")
damole = Unit.create_scaled_unit(mole, "da")
mmole = Unit.create_scaled_unit(mole, "m")
nmole = Unit.create_scaled_unit(mole, "n")
pmole = Unit.create_scaled_unit(mole, "p")
umole = Unit.create_scaled_unit(mole, "u")
Tmole = Unit.create_scaled_unit(mole, "T")
ymole = Unit.create_scaled_unit(mole, "y")
Emole = Unit.create_scaled_unit(mole, "E")
zmole = Unit.create_scaled_unit(mole, "z")
Mmole = Unit.create_scaled_unit(mole, "M")
kmole = Unit.create_scaled_unit(mole, "k")
Ymole = Unit.create_scaled_unit(mole, "Y")
amol = Unit.create_scaled_unit(mol, "a")
cmol = Unit.create_scaled_unit(mol, "c")
Zmol = Unit.create_scaled_unit(mol, "Z")
Pmol = Unit.create_scaled_unit(mol, "P")
dmol = Unit.create_scaled_unit(mol, "d")
Gmol = Unit.create_scaled_unit(mol, "G")
fmol = Unit.create_scaled_unit(mol, "f")
hmol = Unit.create_scaled_unit(mol, "h")
damol = Unit.create_scaled_unit(mol, "da")
mmol = Unit.create_scaled_unit(mol, "m")
nmol = Unit.create_scaled_unit(mol, "n")
pmol = Unit.create_scaled_unit(mol, "p")
umol = Unit.create_scaled_unit(mol, "u")
Tmol = Unit.create_scaled_unit(mol, "T")
ymol = Unit.create_scaled_unit(mol, "y")
Emol = Unit.create_scaled_unit(mol, "E")
zmol = Unit.create_scaled_unit(mol, "z")
Mmol = Unit.create_scaled_unit(mol, "M")
kmol = Unit.create_scaled_unit(mol, "k")
Ymol = Unit.create_scaled_unit(mol, "Y")
acandle = Unit.create_scaled_unit(candle, "a")
ccandle = Unit.create_scaled_unit(candle, "c")
Zcandle = Unit.create_scaled_unit(candle, "Z")
Pcandle = Unit.create_scaled_unit(candle, "P")
dcandle = Unit.create_scaled_unit(candle, "d")
Gcandle = Unit.create_scaled_unit(candle, "G")
fcandle = Unit.create_scaled_unit(candle, "f")
hcandle = Unit.create_scaled_unit(candle, "h")
dacandle = Unit.create_scaled_unit(candle, "da")
mcandle = Unit.create_scaled_unit(candle, "m")
ncandle = Unit.create_scaled_unit(candle, "n")
pcandle = Unit.create_scaled_unit(candle, "p")
ucandle = Unit.create_scaled_unit(candle, "u")
Tcandle = Unit.create_scaled_unit(candle, "T")
ycandle = Unit.create_scaled_unit(candle, "y")
Ecandle = Unit.create_scaled_unit(candle, "E")
zcandle = Unit.create_scaled_unit(candle, "z")
Mcandle = Unit.create_scaled_unit(candle, "M")
kcandle = Unit.create_scaled_unit(candle, "k")
Ycandle = Unit.create_scaled_unit(candle, "Y")
agram = Unit.create_scaled_unit(gram, "a")
cgram = Unit.create_scaled_unit(gram, "c")
Zgram = Unit.create_scaled_unit(gram, "Z")
Pgram = Unit.create_scaled_unit(gram, "P")
dgram = Unit.create_scaled_unit(gram, "d")
Ggram = Unit.create_scaled_unit(gram, "G")
fgram = Unit.create_scaled_unit(gram, "f")
hgram = Unit.create_scaled_unit(gram, "h")
dagram = Unit.create_scaled_unit(gram, "da")
mgram = Unit.create_scaled_unit(gram, "m")
ngram = Unit.create_scaled_unit(gram, "n")
pgram = Unit.create_scaled_unit(gram, "p")
ugram = Unit.create_scaled_unit(gram, "u")
Tgram = Unit.create_scaled_unit(gram, "T")
ygram = Unit.create_scaled_unit(gram, "y")
Egram = Unit.create_scaled_unit(gram, "E")
zgram = Unit.create_scaled_unit(gram, "z")
Mgram = Unit.create_scaled_unit(gram, "M")
kgram = Unit.create_scaled_unit(gram, "k")
Ygram = Unit.create_scaled_unit(gram, "Y")
agramme = Unit.create_scaled_unit(gramme, "a")
cgramme = Unit.create_scaled_unit(gramme, "c")
Zgramme = Unit.create_scaled_unit(gramme, "Z")
Pgramme = Unit.create_scaled_unit(gramme, "P")
dgramme = Unit.create_scaled_unit(gramme, "d")
Ggramme = Unit.create_scaled_unit(gramme, "G")
fgramme = Unit.create_scaled_unit(gramme, "f")
hgramme = Unit.create_scaled_unit(gramme, "h")
dagramme = Unit.create_scaled_unit(gramme, "da")
mgramme = Unit.create_scaled_unit(gramme, "m")
ngramme = Unit.create_scaled_unit(gramme, "n")
pgramme = Unit.create_scaled_unit(gramme, "p")
ugramme = Unit.create_scaled_unit(gramme, "u")
Tgramme = Unit.create_scaled_unit(gramme, "T")
ygramme = Unit.create_scaled_unit(gramme, "y")
Egramme = Unit.create_scaled_unit(gramme, "E")
zgramme = Unit.create_scaled_unit(gramme, "z")
Mgramme = Unit.create_scaled_unit(gramme, "M")
kgramme = Unit.create_scaled_unit(gramme, "k")
Ygramme = Unit.create_scaled_unit(gramme, "Y")
amolar = Unit.create_scaled_unit(molar, "a")
cmolar = Unit.create_scaled_unit(molar, "c")
Zmolar = Unit.create_scaled_unit(molar, "Z")
Pmolar = Unit.create_scaled_unit(molar, "P")
dmolar = Unit.create_scaled_unit(molar, "d")
Gmolar = Unit.create_scaled_unit(molar, "G")
fmolar = Unit.create_scaled_unit(molar, "f")
hmolar = Unit.create_scaled_unit(molar, "h")
damolar = Unit.create_scaled_unit(molar, "da")
mmolar = Unit.create_scaled_unit(molar, "m")
nmolar = Unit.create_scaled_unit(molar, "n")
pmolar = Unit.create_scaled_unit(molar, "p")
umolar = Unit.create_scaled_unit(molar, "u")
Tmolar = Unit.create_scaled_unit(molar, "T")
ymolar = Unit.create_scaled_unit(molar, "y")
Emolar = Unit.create_scaled_unit(molar, "E")
zmolar = Unit.create_scaled_unit(molar, "z")
Mmolar = Unit.create_scaled_unit(molar, "M")
kmolar = Unit.create_scaled_unit(molar, "k")
Ymolar = Unit.create_scaled_unit(molar, "Y")
aradian = Unit.create_scaled_unit(radian, "a")
cradian = Unit.create_scaled_unit(radian, "c")
Zradian = Unit.create_scaled_unit(radian, "Z")
Pradian = Unit.create_scaled_unit(radian, "P")
dradian = Unit.create_scaled_unit(radian, "d")
Gradian = Unit.create_scaled_unit(radian, "G")
fradian = Unit.create_scaled_unit(radian, "f")
hradian = Unit.create_scaled_unit(radian, "h")
daradian = Unit.create_scaled_unit(radian, "da")
mradian = Unit.create_scaled_unit(radian, "m")
nradian = Unit.create_scaled_unit(radian, "n")
pradian = Unit.create_scaled_unit(radian, "p")
uradian = Unit.create_scaled_unit(radian, "u")
Tradian = Unit.create_scaled_unit(radian, "T")
yradian = Unit.create_scaled_unit(radian, "y")
Eradian = Unit.create_scaled_unit(radian, "E")
zradian = Unit.create_scaled_unit(radian, "z")
Mradian = Unit.create_scaled_unit(radian, "M")
kradian = Unit.create_scaled_unit(radian, "k")
Yradian = Unit.create_scaled_unit(radian, "Y")
asteradian = Unit.create_scaled_unit(steradian, "a")
csteradian = Unit.create_scaled_unit(steradian, "c")
Zsteradian = Unit.create_scaled_unit(steradian, "Z")
Psteradian = Unit.create_scaled_unit(steradian, "P")
dsteradian = Unit.create_scaled_unit(steradian, "d")
Gsteradian = Unit.create_scaled_unit(steradian, "G")
fsteradian = Unit.create_scaled_unit(steradian, "f")
hsteradian = Unit.create_scaled_unit(steradian, "h")
dasteradian = Unit.create_scaled_unit(steradian, "da")
msteradian = Unit.create_scaled_unit(steradian, "m")
nsteradian = Unit.create_scaled_unit(steradian, "n")
psteradian = Unit.create_scaled_unit(steradian, "p")
usteradian = Unit.create_scaled_unit(steradian, "u")
Tsteradian = Unit.create_scaled_unit(steradian, "T")
ysteradian = Unit.create_scaled_unit(steradian, "y")
Esteradian = Unit.create_scaled_unit(steradian, "E")
zsteradian = Unit.create_scaled_unit(steradian, "z")
Msteradian = Unit.create_scaled_unit(steradian, "M")
ksteradian = Unit.create_scaled_unit(steradian, "k")
Ysteradian = Unit.create_scaled_unit(steradian, "Y")
ahertz = Unit.create_scaled_unit(hertz, "a")
chertz = Unit.create_scaled_unit(hertz, "c")
Zhertz = Unit.create_scaled_unit(hertz, "Z")
Phertz = Unit.create_scaled_unit(hertz, "P")
dhertz = Unit.create_scaled_unit(hertz, "d")
Ghertz = Unit.create_scaled_unit(hertz, "G")
fhertz = Unit.create_scaled_unit(hertz, "f")
hhertz = Unit.create_scaled_unit(hertz, "h")
dahertz = Unit.create_scaled_unit(hertz, "da")
mhertz = Unit.create_scaled_unit(hertz, "m")
nhertz = Unit.create_scaled_unit(hertz, "n")
phertz = Unit.create_scaled_unit(hertz, "p")
uhertz = Unit.create_scaled_unit(hertz, "u")
Thertz = Unit.create_scaled_unit(hertz, "T")
yhertz = Unit.create_scaled_unit(hertz, "y")
Ehertz = Unit.create_scaled_unit(hertz, "E")
zhertz = Unit.create_scaled_unit(hertz, "z")
Mhertz = Unit.create_scaled_unit(hertz, "M")
khertz = Unit.create_scaled_unit(hertz, "k")
Yhertz = Unit.create_scaled_unit(hertz, "Y")
anewton = Unit.create_scaled_unit(newton, "a")
cnewton = Unit.create_scaled_unit(newton, "c")
Znewton = Unit.create_scaled_unit(newton, "Z")
Pnewton = Unit.create_scaled_unit(newton, "P")
dnewton = Unit.create_scaled_unit(newton, "d")
Gnewton = Unit.create_scaled_unit(newton, "G")
fnewton = Unit.create_scaled_unit(newton, "f")
hnewton = Unit.create_scaled_unit(newton, "h")
danewton = Unit.create_scaled_unit(newton, "da")
mnewton = Unit.create_scaled_unit(newton, "m")
nnewton = Unit.create_scaled_unit(newton, "n")
pnewton = Unit.create_scaled_unit(newton, "p")
unewton = Unit.create_scaled_unit(newton, "u")
Tnewton = Unit.create_scaled_unit(newton, "T")
ynewton = Unit.create_scaled_unit(newton, "y")
Enewton = Unit.create_scaled_unit(newton, "E")
znewton = Unit.create_scaled_unit(newton, "z")
Mnewton = Unit.create_scaled_unit(newton, "M")
knewton = Unit.create_scaled_unit(newton, "k")
Ynewton = Unit.create_scaled_unit(newton, "Y")
apascal = Unit.create_scaled_unit(pascal, "a")
cpascal = Unit.create_scaled_unit(pascal, "c")
Zpascal = Unit.create_scaled_unit(pascal, "Z")
Ppascal = Unit.create_scaled_unit(pascal, "P")
dpascal = Unit.create_scaled_unit(pascal, "d")
Gpascal = Unit.create_scaled_unit(pascal, "G")
fpascal = Unit.create_scaled_unit(pascal, "f")
hpascal = Unit.create_scaled_unit(pascal, "h")
dapascal = Unit.create_scaled_unit(pascal, "da")
mpascal = Unit.create_scaled_unit(pascal, "m")
npascal = Unit.create_scaled_unit(pascal, "n")
ppascal = Unit.create_scaled_unit(pascal, "p")
upascal = Unit.create_scaled_unit(pascal, "u")
Tpascal = Unit.create_scaled_unit(pascal, "T")
ypascal = Unit.create_scaled_unit(pascal, "y")
Epascal = Unit.create_scaled_unit(pascal, "E")
zpascal = Unit.create_scaled_unit(pascal, "z")
Mpascal = Unit.create_scaled_unit(pascal, "M")
kpascal = Unit.create_scaled_unit(pascal, "k")
Ypascal = Unit.create_scaled_unit(pascal, "Y")
ajoule = Unit.create_scaled_unit(joule, "a")
cjoule = Unit.create_scaled_unit(joule, "c")
Zjoule = Unit.create_scaled_unit(joule, "Z")
Pjoule = Unit.create_scaled_unit(joule, "P")
djoule = Unit.create_scaled_unit(joule, "d")
Gjoule = Unit.create_scaled_unit(joule, "G")
fjoule = Unit.create_scaled_unit(joule, "f")
hjoule = Unit.create_scaled_unit(joule, "h")
dajoule = Unit.create_scaled_unit(joule, "da")
mjoule = Unit.create_scaled_unit(joule, "m")
njoule = Unit.create_scaled_unit(joule, "n")
pjoule = Unit.create_scaled_unit(joule, "p")
ujoule = Unit.create_scaled_unit(joule, "u")
Tjoule = Unit.create_scaled_unit(joule, "T")
yjoule = Unit.create_scaled_unit(joule, "y")
Ejoule = Unit.create_scaled_unit(joule, "E")
zjoule = Unit.create_scaled_unit(joule, "z")
Mjoule = Unit.create_scaled_unit(joule, "M")
kjoule = Unit.create_scaled_unit(joule, "k")
Yjoule = Unit.create_scaled_unit(joule, "Y")
awatt = Unit.create_scaled_unit(watt, "a")
cwatt = Unit.create_scaled_unit(watt, "c")
Zwatt = Unit.create_scaled_unit(watt, "Z")
Pwatt = Unit.create_scaled_unit(watt, "P")
dwatt = Unit.create_scaled_unit(watt, "d")
Gwatt = Unit.create_scaled_unit(watt, "G")
fwatt = Unit.create_scaled_unit(watt, "f")
hwatt = Unit.create_scaled_unit(watt, "h")
dawatt = Unit.create_scaled_unit(watt, "da")
mwatt = Unit.create_scaled_unit(watt, "m")
nwatt = Unit.create_scaled_unit(watt, "n")
pwatt = Unit.create_scaled_unit(watt, "p")
uwatt = Unit.create_scaled_unit(watt, "u")
Twatt = Unit.create_scaled_unit(watt, "T")
ywatt = Unit.create_scaled_unit(watt, "y")
Ewatt = Unit.create_scaled_unit(watt, "E")
zwatt = Unit.create_scaled_unit(watt, "z")
Mwatt = Unit.create_scaled_unit(watt, "M")
kwatt = Unit.create_scaled_unit(watt, "k")
Ywatt = Unit.create_scaled_unit(watt, "Y")
acoulomb = Unit.create_scaled_unit(coulomb, "a")
ccoulomb = Unit.create_scaled_unit(coulomb, "c")
Zcoulomb = Unit.create_scaled_unit(coulomb, "Z")
Pcoulomb = Unit.create_scaled_unit(coulomb, "P")
dcoulomb = Unit.create_scaled_unit(coulomb, "d")
Gcoulomb = Unit.create_scaled_unit(coulomb, "G")
fcoulomb = Unit.create_scaled_unit(coulomb, "f")
hcoulomb = Unit.create_scaled_unit(coulomb, "h")
dacoulomb = Unit.create_scaled_unit(coulomb, "da")
mcoulomb = Unit.create_scaled_unit(coulomb, "m")
ncoulomb = Unit.create_scaled_unit(coulomb, "n")
pcoulomb = Unit.create_scaled_unit(coulomb, "p")
ucoulomb = Unit.create_scaled_unit(coulomb, "u")
Tcoulomb = Unit.create_scaled_unit(coulomb, "T")
ycoulomb = Unit.create_scaled_unit(coulomb, "y")
Ecoulomb = Unit.create_scaled_unit(coulomb, "E")
zcoulomb = Unit.create_scaled_unit(coulomb, "z")
Mcoulomb = Unit.create_scaled_unit(coulomb, "M")
kcoulomb = Unit.create_scaled_unit(coulomb, "k")
Ycoulomb = Unit.create_scaled_unit(coulomb, "Y")
avolt = Unit.create_scaled_unit(volt, "a")
cvolt = Unit.create_scaled_unit(volt, "c")
Zvolt = Unit.create_scaled_unit(volt, "Z")
Pvolt = Unit.create_scaled_unit(volt, "P")
dvolt = Unit.create_scaled_unit(volt, "d")
Gvolt = Unit.create_scaled_unit(volt, "G")
fvolt = Unit.create_scaled_unit(volt, "f")
hvolt = Unit.create_scaled_unit(volt, "h")
davolt = Unit.create_scaled_unit(volt, "da")
mvolt = Unit.create_scaled_unit(volt, "m")
nvolt = Unit.create_scaled_unit(volt, "n")
pvolt = Unit.create_scaled_unit(volt, "p")
uvolt = Unit.create_scaled_unit(volt, "u")
Tvolt = Unit.create_scaled_unit(volt, "T")
yvolt = Unit.create_scaled_unit(volt, "y")
Evolt = Unit.create_scaled_unit(volt, "E")
zvolt = Unit.create_scaled_unit(volt, "z")
Mvolt = Unit.create_scaled_unit(volt, "M")
kvolt = Unit.create_scaled_unit(volt, "k")
Yvolt = Unit.create_scaled_unit(volt, "Y")
afarad = Unit.create_scaled_unit(farad, "a")
cfarad = Unit.create_scaled_unit(farad, "c")
Zfarad = Unit.create_scaled_unit(farad, "Z")
Pfarad = Unit.create_scaled_unit(farad, "P")
dfarad = Unit.create_scaled_unit(farad, "d")
Gfarad = Unit.create_scaled_unit(farad, "G")
ffarad = Unit.create_scaled_unit(farad, "f")
hfarad = Unit.create_scaled_unit(farad, "h")
dafarad = Unit.create_scaled_unit(farad, "da")
mfarad = Unit.create_scaled_unit(farad, "m")
nfarad = Unit.create_scaled_unit(farad, "n")
pfarad = Unit.create_scaled_unit(farad, "p")
ufarad = Unit.create_scaled_unit(farad, "u")
Tfarad = Unit.create_scaled_unit(farad, "T")
yfarad = Unit.create_scaled_unit(farad, "y")
Efarad = Unit.create_scaled_unit(farad, "E")
zfarad = Unit.create_scaled_unit(farad, "z")
Mfarad = Unit.create_scaled_unit(farad, "M")
kfarad = Unit.create_scaled_unit(farad, "k")
Yfarad = Unit.create_scaled_unit(farad, "Y")
aohm = Unit.create_scaled_unit(ohm, "a")
cohm = Unit.create_scaled_unit(ohm, "c")
Zohm = Unit.create_scaled_unit(ohm, "Z")
Pohm = Unit.create_scaled_unit(ohm, "P")
dohm = Unit.create_scaled_unit(ohm, "d")
Gohm = Unit.create_scaled_unit(ohm, "G")
fohm = Unit.create_scaled_unit(ohm, "f")
hohm = Unit.create_scaled_unit(ohm, "h")
daohm = Unit.create_scaled_unit(ohm, "da")
mohm = Unit.create_scaled_unit(ohm, "m")
nohm = Unit.create_scaled_unit(ohm, "n")
pohm = Unit.create_scaled_unit(ohm, "p")
uohm = Unit.create_scaled_unit(ohm, "u")
Tohm = Unit.create_scaled_unit(ohm, "T")
yohm = Unit.create_scaled_unit(ohm, "y")
Eohm = Unit.create_scaled_unit(ohm, "E")
zohm = Unit.create_scaled_unit(ohm, "z")
Mohm = Unit.create_scaled_unit(ohm, "M")
kohm = Unit.create_scaled_unit(ohm, "k")
Yohm = Unit.create_scaled_unit(ohm, "Y")
asiemens = Unit.create_scaled_unit(siemens, "a")
csiemens = Unit.create_scaled_unit(siemens, "c")
Zsiemens = Unit.create_scaled_unit(siemens, "Z")
Psiemens = Unit.create_scaled_unit(siemens, "P")
dsiemens = Unit.create_scaled_unit(siemens, "d")
Gsiemens = Unit.create_scaled_unit(siemens, "G")
fsiemens = Unit.create_scaled_unit(siemens, "f")
hsiemens = Unit.create_scaled_unit(siemens, "h")
dasiemens = Unit.create_scaled_unit(siemens, "da")
msiemens = Unit.create_scaled_unit(siemens, "m")
nsiemens = Unit.create_scaled_unit(siemens, "n")
psiemens = Unit.create_scaled_unit(siemens, "p")
usiemens = Unit.create_scaled_unit(siemens, "u")
Tsiemens = Unit.create_scaled_unit(siemens, "T")
ysiemens = Unit.create_scaled_unit(siemens, "y")
Esiemens = Unit.create_scaled_unit(siemens, "E")
zsiemens = Unit.create_scaled_unit(siemens, "z")
Msiemens = Unit.create_scaled_unit(siemens, "M")
ksiemens = Unit.create_scaled_unit(siemens, "k")
Ysiemens = Unit.create_scaled_unit(siemens, "Y")
aweber = Unit.create_scaled_unit(weber, "a")
cweber = Unit.create_scaled_unit(weber, "c")
Zweber = Unit.create_scaled_unit(weber, "Z")
Pweber = Unit.create_scaled_unit(weber, "P")
dweber = Unit.create_scaled_unit(weber, "d")
Gweber = Unit.create_scaled_unit(weber, "G")
fweber = Unit.create_scaled_unit(weber, "f")
hweber = Unit.create_scaled_unit(weber, "h")
daweber = Unit.create_scaled_unit(weber, "da")
mweber = Unit.create_scaled_unit(weber, "m")
nweber = Unit.create_scaled_unit(weber, "n")
pweber = Unit.create_scaled_unit(weber, "p")
uweber = Unit.create_scaled_unit(weber, "u")
Tweber = Unit.create_scaled_unit(weber, "T")
yweber = Unit.create_scaled_unit(weber, "y")
Eweber = Unit.create_scaled_unit(weber, "E")
zweber = Unit.create_scaled_unit(weber, "z")
Mweber = Unit.create_scaled_unit(weber, "M")
kweber = Unit.create_scaled_unit(weber, "k")
Yweber = Unit.create_scaled_unit(weber, "Y")
atesla = Unit.create_scaled_unit(tesla, "a")
ctesla = Unit.create_scaled_unit(tesla, "c")
Ztesla = Unit.create_scaled_unit(tesla, "Z")
Ptesla = Unit.create_scaled_unit(tesla, "P")
dtesla = Unit.create_scaled_unit(tesla, "d")
Gtesla = Unit.create_scaled_unit(tesla, "G")
ftesla = Unit.create_scaled_unit(tesla, "f")
htesla = Unit.create_scaled_unit(tesla, "h")
datesla = Unit.create_scaled_unit(tesla, "da")
mtesla = Unit.create_scaled_unit(tesla, "m")
ntesla = Unit.create_scaled_unit(tesla, "n")
ptesla = Unit.create_scaled_unit(tesla, "p")
utesla = Unit.create_scaled_unit(tesla, "u")
Ttesla = Unit.create_scaled_unit(tesla, "T")
ytesla = Unit.create_scaled_unit(tesla, "y")
Etesla = Unit.create_scaled_unit(tesla, "E")
ztesla = Unit.create_scaled_unit(tesla, "z")
Mtesla = Unit.create_scaled_unit(tesla, "M")
ktesla = Unit.create_scaled_unit(tesla, "k")
Ytesla = Unit.create_scaled_unit(tesla, "Y")
ahenry = Unit.create_scaled_unit(henry, "a")
chenry = Unit.create_scaled_unit(henry, "c")
Zhenry = Unit.create_scaled_unit(henry, "Z")
Phenry = Unit.create_scaled_unit(henry, "P")
dhenry = Unit.create_scaled_unit(henry, "d")
Ghenry = Unit.create_scaled_unit(henry, "G")
fhenry = Unit.create_scaled_unit(henry, "f")
hhenry = Unit.create_scaled_unit(henry, "h")
dahenry = Unit.create_scaled_unit(henry, "da")
mhenry = Unit.create_scaled_unit(henry, "m")
nhenry = Unit.create_scaled_unit(henry, "n")
phenry = Unit.create_scaled_unit(henry, "p")
uhenry = Unit.create_scaled_unit(henry, "u")
Thenry = Unit.create_scaled_unit(henry, "T")
yhenry = Unit.create_scaled_unit(henry, "y")
Ehenry = Unit.create_scaled_unit(henry, "E")
zhenry = Unit.create_scaled_unit(henry, "z")
Mhenry = Unit.create_scaled_unit(henry, "M")
khenry = Unit.create_scaled_unit(henry, "k")
Yhenry = Unit.create_scaled_unit(henry, "Y")
alumen = Unit.create_scaled_unit(lumen, "a")
clumen = Unit.create_scaled_unit(lumen, "c")
Zlumen = Unit.create_scaled_unit(lumen, "Z")
Plumen = Unit.create_scaled_unit(lumen, "P")
dlumen = Unit.create_scaled_unit(lumen, "d")
Glumen = Unit.create_scaled_unit(lumen, "G")
flumen = Unit.create_scaled_unit(lumen, "f")
hlumen = Unit.create_scaled_unit(lumen, "h")
dalumen = Unit.create_scaled_unit(lumen, "da")
mlumen = Unit.create_scaled_unit(lumen, "m")
nlumen = Unit.create_scaled_unit(lumen, "n")
plumen = Unit.create_scaled_unit(lumen, "p")
ulumen = Unit.create_scaled_unit(lumen, "u")
Tlumen = Unit.create_scaled_unit(lumen, "T")
ylumen = Unit.create_scaled_unit(lumen, "y")
Elumen = Unit.create_scaled_unit(lumen, "E")
zlumen = Unit.create_scaled_unit(lumen, "z")
Mlumen = Unit.create_scaled_unit(lumen, "M")
klumen = Unit.create_scaled_unit(lumen, "k")
Ylumen = Unit.create_scaled_unit(lumen, "Y")
alux = Unit.create_scaled_unit(lux, "a")
clux = Unit.create_scaled_unit(lux, "c")
Zlux = Unit.create_scaled_unit(lux, "Z")
Plux = Unit.create_scaled_unit(lux, "P")
dlux = Unit.create_scaled_unit(lux, "d")
Glux = Unit.create_scaled_unit(lux, "G")
flux = Unit.create_scaled_unit(lux, "f")
hlux = Unit.create_scaled_unit(lux, "h")
dalux = Unit.create_scaled_unit(lux, "da")
mlux = Unit.create_scaled_unit(lux, "m")
nlux = Unit.create_scaled_unit(lux, "n")
plux = Unit.create_scaled_unit(lux, "p")
ulux = Unit.create_scaled_unit(lux, "u")
Tlux = Unit.create_scaled_unit(lux, "T")
ylux = Unit.create_scaled_unit(lux, "y")
Elux = Unit.create_scaled_unit(lux, "E")
zlux = Unit.create_scaled_unit(lux, "z")
Mlux = Unit.create_scaled_unit(lux, "M")
klux = Unit.create_scaled_unit(lux, "k")
Ylux = Unit.create_scaled_unit(lux, "Y")
abecquerel = Unit.create_scaled_unit(becquerel, "a")
cbecquerel = Unit.create_scaled_unit(becquerel, "c")
Zbecquerel = Unit.create_scaled_unit(becquerel, "Z")
Pbecquerel = Unit.create_scaled_unit(becquerel, "P")
dbecquerel = Unit.create_scaled_unit(becquerel, "d")
Gbecquerel = Unit.create_scaled_unit(becquerel, "G")
fbecquerel = Unit.create_scaled_unit(becquerel, "f")
hbecquerel = Unit.create_scaled_unit(becquerel, "h")
dabecquerel = Unit.create_scaled_unit(becquerel, "da")
mbecquerel = Unit.create_scaled_unit(becquerel, "m")
nbecquerel = Unit.create_scaled_unit(becquerel, "n")
pbecquerel = Unit.create_scaled_unit(becquerel, "p")
ubecquerel = Unit.create_scaled_unit(becquerel, "u")
Tbecquerel = Unit.create_scaled_unit(becquerel, "T")
ybecquerel = Unit.create_scaled_unit(becquerel, "y")
Ebecquerel = Unit.create_scaled_unit(becquerel, "E")
zbecquerel = Unit.create_scaled_unit(becquerel, "z")
Mbecquerel = Unit.create_scaled_unit(becquerel, "M")
kbecquerel = Unit.create_scaled_unit(becquerel, "k")
Ybecquerel = Unit.create_scaled_unit(becquerel, "Y")
agray = Unit.create_scaled_unit(gray, "a")
cgray = Unit.create_scaled_unit(gray, "c")
Zgray = Unit.create_scaled_unit(gray, "Z")
Pgray = Unit.create_scaled_unit(gray, "P")
dgray = Unit.create_scaled_unit(gray, "d")
Ggray = Unit.create_scaled_unit(gray, "G")
fgray = Unit.create_scaled_unit(gray, "f")
hgray = Unit.create_scaled_unit(gray, "h")
dagray = Unit.create_scaled_unit(gray, "da")
mgray = Unit.create_scaled_unit(gray, "m")
ngray = Unit.create_scaled_unit(gray, "n")
pgray = Unit.create_scaled_unit(gray, "p")
ugray = Unit.create_scaled_unit(gray, "u")
Tgray = Unit.create_scaled_unit(gray, "T")
ygray = Unit.create_scaled_unit(gray, "y")
Egray = Unit.create_scaled_unit(gray, "E")
zgray = Unit.create_scaled_unit(gray, "z")
Mgray = Unit.create_scaled_unit(gray, "M")
kgray = Unit.create_scaled_unit(gray, "k")
Ygray = Unit.create_scaled_unit(gray, "Y")
asievert = Unit.create_scaled_unit(sievert, "a")
csievert = Unit.create_scaled_unit(sievert, "c")
Zsievert = Unit.create_scaled_unit(sievert, "Z")
Psievert = Unit.create_scaled_unit(sievert, "P")
dsievert = Unit.create_scaled_unit(sievert, "d")
Gsievert = Unit.create_scaled_unit(sievert, "G")
fsievert = Unit.create_scaled_unit(sievert, "f")
hsievert = Unit.create_scaled_unit(sievert, "h")
dasievert = Unit.create_scaled_unit(sievert, "da")
msievert = Unit.create_scaled_unit(sievert, "m")
nsievert = Unit.create_scaled_unit(sievert, "n")
psievert = Unit.create_scaled_unit(sievert, "p")
usievert = Unit.create_scaled_unit(sievert, "u")
Tsievert = Unit.create_scaled_unit(sievert, "T")
ysievert = Unit.create_scaled_unit(sievert, "y")
Esievert = Unit.create_scaled_unit(sievert, "E")
zsievert = Unit.create_scaled_unit(sievert, "z")
Msievert = Unit.create_scaled_unit(sievert, "M")
ksievert = Unit.create_scaled_unit(sievert, "k")
Ysievert = Unit.create_scaled_unit(sievert, "Y")
akatal = Unit.create_scaled_unit(katal, "a")
ckatal = Unit.create_scaled_unit(katal, "c")
Zkatal = Unit.create_scaled_unit(katal, "Z")
Pkatal = Unit.create_scaled_unit(katal, "P")
dkatal = Unit.create_scaled_unit(katal, "d")
Gkatal = Unit.create_scaled_unit(katal, "G")
fkatal = Unit.create_scaled_unit(katal, "f")
hkatal = Unit.create_scaled_unit(katal, "h")
dakatal = Unit.create_scaled_unit(katal, "da")
mkatal = Unit.create_scaled_unit(katal, "m")
nkatal = Unit.create_scaled_unit(katal, "n")
pkatal = Unit.create_scaled_unit(katal, "p")
ukatal = Unit.create_scaled_unit(katal, "u")
Tkatal = Unit.create_scaled_unit(katal, "T")
ykatal = Unit.create_scaled_unit(katal, "y")
Ekatal = Unit.create_scaled_unit(katal, "E")
zkatal = Unit.create_scaled_unit(katal, "z")
Mkatal = Unit.create_scaled_unit(katal, "M")
kkatal = Unit.create_scaled_unit(katal, "k")
Ykatal = Unit.create_scaled_unit(katal, "Y")
######### SCALED BASE UNITS TO POWERS ###########
metre2 = Unit.create((metre ** 2).dim, name="metre2", dispname=f"{str(metre)}^2", scale=metre.scale * 2)
metre3 = Unit.create((metre ** 3).dim, name="metre3", dispname=f"{str(metre)}^3", scale=metre.scale * 3)
meter2 = Unit.create((meter ** 2).dim, name="meter2", dispname=f"{str(meter)}^2", scale=meter.scale * 2)
meter3 = Unit.create((meter ** 3).dim, name="meter3", dispname=f"{str(meter)}^3", scale=meter.scale * 3)
kilogram2 = Unit.create((kilogram ** 2).dim, name="kilogram2", dispname=f"{str(kilogram)}^2",
                        scale=kilogram.scale * 2)
kilogram3 = Unit.create((kilogram ** 3).dim, name="kilogram3", dispname=f"{str(kilogram)}^3",
                        scale=kilogram.scale * 3)
second2 = Unit.create((second ** 2).dim, name="second2", dispname=f"{str(second)}^2", scale=second.scale * 2)
second3 = Unit.create((second ** 3).dim, name="second3", dispname=f"{str(second)}^3", scale=second.scale * 3)
amp2 = Unit.create((amp ** 2).dim, name="amp2", dispname=f"{str(amp)}^2", scale=amp.scale * 2)
amp3 = Unit.create((amp ** 3).dim, name="amp3", dispname=f"{str(amp)}^3", scale=amp.scale * 3)
ampere2 = Unit.create((ampere ** 2).dim, name="ampere2", dispname=f"{str(ampere)}^2", scale=ampere.scale * 2)
ampere3 = Unit.create((ampere ** 3).dim, name="ampere3", dispname=f"{str(ampere)}^3", scale=ampere.scale * 3)
kelvin2 = Unit.create((kelvin ** 2).dim, name="kelvin2", dispname=f"{str(kelvin)}^2", scale=kelvin.scale * 2)
kelvin3 = Unit.create((kelvin ** 3).dim, name="kelvin3", dispname=f"{str(kelvin)}^3", scale=kelvin.scale * 3)
mole2 = Unit.create((mole ** 2).dim, name="mole2", dispname=f"{str(mole)}^2", scale=mole.scale * 2)
mole3 = Unit.create((mole ** 3).dim, name="mole3", dispname=f"{str(mole)}^3", scale=mole.scale * 3)
mol2 = Unit.create((mol ** 2).dim, name="mol2", dispname=f"{str(mol)}^2", scale=mol.scale * 2)
mol3 = Unit.create((mol ** 3).dim, name="mol3", dispname=f"{str(mol)}^3", scale=mol.scale * 3)
candle2 = Unit.create((candle ** 2).dim, name="candle2", dispname=f"{str(candle)}^2", scale=candle.scale * 2)
candle3 = Unit.create((candle ** 3).dim, name="candle3", dispname=f"{str(candle)}^3", scale=candle.scale * 3)
kilogramme2 = Unit.create((kilogramme ** 2).dim, name="kilogramme2", dispname=f"{str(kilogramme)}^2",
                          scale=kilogramme.scale * 2)
kilogramme3 = Unit.create((kilogramme ** 3).dim, name="kilogramme3", dispname=f"{str(kilogramme)}^3",
                          scale=kilogramme.scale * 3)
gram2 = Unit.create((gram ** 2).dim, name="gram2", dispname=f"{str(gram)}^2", scale=gram.scale * 2)
gram3 = Unit.create((gram ** 3).dim, name="gram3", dispname=f"{str(gram)}^3", scale=gram.scale * 3)
gramme2 = Unit.create((gramme ** 2).dim, name="gramme2", dispname=f"{str(gramme)}^2", scale=gramme.scale * 2)
gramme3 = Unit.create((gramme ** 3).dim, name="gramme3", dispname=f"{str(gramme)}^3", scale=gramme.scale * 3)
molar2 = Unit.create((molar ** 2).dim, name="molar2", dispname=f"{str(molar)}^2", scale=molar.scale * 2)
molar3 = Unit.create((molar ** 3).dim, name="molar3", dispname=f"{str(molar)}^3", scale=molar.scale * 3)
radian2 = Unit.create((radian ** 2).dim, name="radian2", dispname=f"{str(radian)}^2", scale=radian.scale * 2)
radian3 = Unit.create((radian ** 3).dim, name="radian3", dispname=f"{str(radian)}^3", scale=radian.scale * 3)
steradian2 = Unit.create((steradian ** 2).dim, name="steradian2", dispname=f"{str(steradian)}^2",
                         scale=steradian.scale * 2)
steradian3 = Unit.create((steradian ** 3).dim, name="steradian3", dispname=f"{str(steradian)}^3",
                         scale=steradian.scale * 3)
hertz2 = Unit.create((hertz ** 2).dim, name="hertz2", dispname=f"{str(hertz)}^2", scale=hertz.scale * 2)
hertz3 = Unit.create((hertz ** 3).dim, name="hertz3", dispname=f"{str(hertz)}^3", scale=hertz.scale * 3)
newton2 = Unit.create((newton ** 2).dim, name="newton2", dispname=f"{str(newton)}^2", scale=newton.scale * 2)
newton3 = Unit.create((newton ** 3).dim, name="newton3", dispname=f"{str(newton)}^3", scale=newton.scale * 3)
pascal2 = Unit.create((pascal ** 2).dim, name="pascal2", dispname=f"{str(pascal)}^2", scale=pascal.scale * 2)
pascal3 = Unit.create((pascal ** 3).dim, name="pascal3", dispname=f"{str(pascal)}^3", scale=pascal.scale * 3)
joule2 = Unit.create((joule ** 2).dim, name="joule2", dispname=f"{str(joule)}^2", scale=joule.scale * 2)
joule3 = Unit.create((joule ** 3).dim, name="joule3", dispname=f"{str(joule)}^3", scale=joule.scale * 3)
watt2 = Unit.create((watt ** 2).dim, name="watt2", dispname=f"{str(watt)}^2", scale=watt.scale * 2)
watt3 = Unit.create((watt ** 3).dim, name="watt3", dispname=f"{str(watt)}^3", scale=watt.scale * 3)
coulomb2 = Unit.create((coulomb ** 2).dim, name="coulomb2", dispname=f"{str(coulomb)}^2", scale=coulomb.scale * 2)
coulomb3 = Unit.create((coulomb ** 3).dim, name="coulomb3", dispname=f"{str(coulomb)}^3", scale=coulomb.scale * 3)
volt2 = Unit.create((volt ** 2).dim, name="volt2", dispname=f"{str(volt)}^2", scale=volt.scale * 2)
volt3 = Unit.create((volt ** 3).dim, name="volt3", dispname=f"{str(volt)}^3", scale=volt.scale * 3)
farad2 = Unit.create((farad ** 2).dim, name="farad2", dispname=f"{str(farad)}^2", scale=farad.scale * 2)
farad3 = Unit.create((farad ** 3).dim, name="farad3", dispname=f"{str(farad)}^3", scale=farad.scale * 3)
ohm2 = Unit.create((ohm ** 2).dim, name="ohm2", dispname=f"{str(ohm)}^2", scale=ohm.scale * 2)
ohm3 = Unit.create((ohm ** 3).dim, name="ohm3", dispname=f"{str(ohm)}^3", scale=ohm.scale * 3)
siemens2 = Unit.create((siemens ** 2).dim, name="siemens2", dispname=f"{str(siemens)}^2", scale=siemens.scale * 2)
siemens3 = Unit.create((siemens ** 3).dim, name="siemens3", dispname=f"{str(siemens)}^3", scale=siemens.scale * 3)
weber2 = Unit.create((weber ** 2).dim, name="weber2", dispname=f"{str(weber)}^2", scale=weber.scale * 2)
weber3 = Unit.create((weber ** 3).dim, name="weber3", dispname=f"{str(weber)}^3", scale=weber.scale * 3)
tesla2 = Unit.create((tesla ** 2).dim, name="tesla2", dispname=f"{str(tesla)}^2", scale=tesla.scale * 2)
tesla3 = Unit.create((tesla ** 3).dim, name="tesla3", dispname=f"{str(tesla)}^3", scale=tesla.scale * 3)
henry2 = Unit.create((henry ** 2).dim, name="henry2", dispname=f"{str(henry)}^2", scale=henry.scale * 2)
henry3 = Unit.create((henry ** 3).dim, name="henry3", dispname=f"{str(henry)}^3", scale=henry.scale * 3)
lumen2 = Unit.create((lumen ** 2).dim, name="lumen2", dispname=f"{str(lumen)}^2", scale=lumen.scale * 2)
lumen3 = Unit.create((lumen ** 3).dim, name="lumen3", dispname=f"{str(lumen)}^3", scale=lumen.scale * 3)
lux2 = Unit.create((lux ** 2).dim, name="lux2", dispname=f"{str(lux)}^2", scale=lux.scale * 2)
lux3 = Unit.create((lux ** 3).dim, name="lux3", dispname=f"{str(lux)}^3", scale=lux.scale * 3)
becquerel2 = Unit.create((becquerel ** 2).dim, name="becquerel2", dispname=f"{str(becquerel)}^2",
                         scale=becquerel.scale * 2)
becquerel3 = Unit.create((becquerel ** 3).dim, name="becquerel3", dispname=f"{str(becquerel)}^3",
                         scale=becquerel.scale * 3)
gray2 = Unit.create((gray ** 2).dim, name="gray2", dispname=f"{str(gray)}^2", scale=gray.scale * 2)
gray3 = Unit.create((gray ** 3).dim, name="gray3", dispname=f"{str(gray)}^3", scale=gray.scale * 3)
sievert2 = Unit.create((sievert ** 2).dim, name="sievert2", dispname=f"{str(sievert)}^2", scale=sievert.scale * 2)
sievert3 = Unit.create((sievert ** 3).dim, name="sievert3", dispname=f"{str(sievert)}^3", scale=sievert.scale * 3)
katal2 = Unit.create((katal ** 2).dim, name="katal2", dispname=f"{str(katal)}^2", scale=katal.scale * 2)
katal3 = Unit.create((katal ** 3).dim, name="katal3", dispname=f"{str(katal)}^3", scale=katal.scale * 3)
ametre2 = Unit.create((ametre ** 2).dim, name="ametre2", dispname=f"{str(ametre)}^2", scale=ametre.scale * 2)
ametre3 = Unit.create((ametre ** 3).dim, name="ametre3", dispname=f"{str(ametre)}^3", scale=ametre.scale * 3)
cmetre2 = Unit.create((cmetre ** 2).dim, name="cmetre2", dispname=f"{str(cmetre)}^2", scale=cmetre.scale * 2)
cmetre3 = Unit.create((cmetre ** 3).dim, name="cmetre3", dispname=f"{str(cmetre)}^3", scale=cmetre.scale * 3)
Zmetre2 = Unit.create((Zmetre ** 2).dim, name="Zmetre2", dispname=f"{str(Zmetre)}^2", scale=Zmetre.scale * 2)
Zmetre3 = Unit.create((Zmetre ** 3).dim, name="Zmetre3", dispname=f"{str(Zmetre)}^3", scale=Zmetre.scale * 3)
Pmetre2 = Unit.create((Pmetre ** 2).dim, name="Pmetre2", dispname=f"{str(Pmetre)}^2", scale=Pmetre.scale * 2)
Pmetre3 = Unit.create((Pmetre ** 3).dim, name="Pmetre3", dispname=f"{str(Pmetre)}^3", scale=Pmetre.scale * 3)
dmetre2 = Unit.create((dmetre ** 2).dim, name="dmetre2", dispname=f"{str(dmetre)}^2", scale=dmetre.scale * 2)
dmetre3 = Unit.create((dmetre ** 3).dim, name="dmetre3", dispname=f"{str(dmetre)}^3", scale=dmetre.scale * 3)
Gmetre2 = Unit.create((Gmetre ** 2).dim, name="Gmetre2", dispname=f"{str(Gmetre)}^2", scale=Gmetre.scale * 2)
Gmetre3 = Unit.create((Gmetre ** 3).dim, name="Gmetre3", dispname=f"{str(Gmetre)}^3", scale=Gmetre.scale * 3)
fmetre2 = Unit.create((fmetre ** 2).dim, name="fmetre2", dispname=f"{str(fmetre)}^2", scale=fmetre.scale * 2)
fmetre3 = Unit.create((fmetre ** 3).dim, name="fmetre3", dispname=f"{str(fmetre)}^3", scale=fmetre.scale * 3)
hmetre2 = Unit.create((hmetre ** 2).dim, name="hmetre2", dispname=f"{str(hmetre)}^2", scale=hmetre.scale * 2)
hmetre3 = Unit.create((hmetre ** 3).dim, name="hmetre3", dispname=f"{str(hmetre)}^3", scale=hmetre.scale * 3)
dametre2 = Unit.create((dametre ** 2).dim, name="dametre2", dispname=f"{str(dametre)}^2", scale=dametre.scale * 2)
dametre3 = Unit.create((dametre ** 3).dim, name="dametre3", dispname=f"{str(dametre)}^3", scale=dametre.scale * 3)
mmetre2 = Unit.create((mmetre ** 2).dim, name="mmetre2", dispname=f"{str(mmetre)}^2", scale=mmetre.scale * 2)
mmetre3 = Unit.create((mmetre ** 3).dim, name="mmetre3", dispname=f"{str(mmetre)}^3", scale=mmetre.scale * 3)
nmetre2 = Unit.create((nmetre ** 2).dim, name="nmetre2", dispname=f"{str(nmetre)}^2", scale=nmetre.scale * 2)
nmetre3 = Unit.create((nmetre ** 3).dim, name="nmetre3", dispname=f"{str(nmetre)}^3", scale=nmetre.scale * 3)
pmetre2 = Unit.create((pmetre ** 2).dim, name="pmetre2", dispname=f"{str(pmetre)}^2", scale=pmetre.scale * 2)
pmetre3 = Unit.create((pmetre ** 3).dim, name="pmetre3", dispname=f"{str(pmetre)}^3", scale=pmetre.scale * 3)
umetre2 = Unit.create((umetre ** 2).dim, name="umetre2", dispname=f"{str(umetre)}^2", scale=umetre.scale * 2)
umetre3 = Unit.create((umetre ** 3).dim, name="umetre3", dispname=f"{str(umetre)}^3", scale=umetre.scale * 3)
Tmetre2 = Unit.create((Tmetre ** 2).dim, name="Tmetre2", dispname=f"{str(Tmetre)}^2", scale=Tmetre.scale * 2)
Tmetre3 = Unit.create((Tmetre ** 3).dim, name="Tmetre3", dispname=f"{str(Tmetre)}^3", scale=Tmetre.scale * 3)
ymetre2 = Unit.create((ymetre ** 2).dim, name="ymetre2", dispname=f"{str(ymetre)}^2", scale=ymetre.scale * 2)
ymetre3 = Unit.create((ymetre ** 3).dim, name="ymetre3", dispname=f"{str(ymetre)}^3", scale=ymetre.scale * 3)
Emetre2 = Unit.create((Emetre ** 2).dim, name="Emetre2", dispname=f"{str(Emetre)}^2", scale=Emetre.scale * 2)
Emetre3 = Unit.create((Emetre ** 3).dim, name="Emetre3", dispname=f"{str(Emetre)}^3", scale=Emetre.scale * 3)
zmetre2 = Unit.create((zmetre ** 2).dim, name="zmetre2", dispname=f"{str(zmetre)}^2", scale=zmetre.scale * 2)
zmetre3 = Unit.create((zmetre ** 3).dim, name="zmetre3", dispname=f"{str(zmetre)}^3", scale=zmetre.scale * 3)
Mmetre2 = Unit.create((Mmetre ** 2).dim, name="Mmetre2", dispname=f"{str(Mmetre)}^2", scale=Mmetre.scale * 2)
Mmetre3 = Unit.create((Mmetre ** 3).dim, name="Mmetre3", dispname=f"{str(Mmetre)}^3", scale=Mmetre.scale * 3)
kmetre2 = Unit.create((kmetre ** 2).dim, name="kmetre2", dispname=f"{str(kmetre)}^2", scale=kmetre.scale * 2)
kmetre3 = Unit.create((kmetre ** 3).dim, name="kmetre3", dispname=f"{str(kmetre)}^3", scale=kmetre.scale * 3)
Ymetre2 = Unit.create((Ymetre ** 2).dim, name="Ymetre2", dispname=f"{str(Ymetre)}^2", scale=Ymetre.scale * 2)
Ymetre3 = Unit.create((Ymetre ** 3).dim, name="Ymetre3", dispname=f"{str(Ymetre)}^3", scale=Ymetre.scale * 3)
ameter2 = Unit.create((ameter ** 2).dim, name="ameter2", dispname=f"{str(ameter)}^2", scale=ameter.scale * 2)
ameter3 = Unit.create((ameter ** 3).dim, name="ameter3", dispname=f"{str(ameter)}^3", scale=ameter.scale * 3)
cmeter2 = Unit.create((cmeter ** 2).dim, name="cmeter2", dispname=f"{str(cmeter)}^2", scale=cmeter.scale * 2)
cmeter3 = Unit.create((cmeter ** 3).dim, name="cmeter3", dispname=f"{str(cmeter)}^3", scale=cmeter.scale * 3)
Zmeter2 = Unit.create((Zmeter ** 2).dim, name="Zmeter2", dispname=f"{str(Zmeter)}^2", scale=Zmeter.scale * 2)
Zmeter3 = Unit.create((Zmeter ** 3).dim, name="Zmeter3", dispname=f"{str(Zmeter)}^3", scale=Zmeter.scale * 3)
Pmeter2 = Unit.create((Pmeter ** 2).dim, name="Pmeter2", dispname=f"{str(Pmeter)}^2", scale=Pmeter.scale * 2)
Pmeter3 = Unit.create((Pmeter ** 3).dim, name="Pmeter3", dispname=f"{str(Pmeter)}^3", scale=Pmeter.scale * 3)
dmeter2 = Unit.create((dmeter ** 2).dim, name="dmeter2", dispname=f"{str(dmeter)}^2", scale=dmeter.scale * 2)
dmeter3 = Unit.create((dmeter ** 3).dim, name="dmeter3", dispname=f"{str(dmeter)}^3", scale=dmeter.scale * 3)
Gmeter2 = Unit.create((Gmeter ** 2).dim, name="Gmeter2", dispname=f"{str(Gmeter)}^2", scale=Gmeter.scale * 2)
Gmeter3 = Unit.create((Gmeter ** 3).dim, name="Gmeter3", dispname=f"{str(Gmeter)}^3", scale=Gmeter.scale * 3)
fmeter2 = Unit.create((fmeter ** 2).dim, name="fmeter2", dispname=f"{str(fmeter)}^2", scale=fmeter.scale * 2)
fmeter3 = Unit.create((fmeter ** 3).dim, name="fmeter3", dispname=f"{str(fmeter)}^3", scale=fmeter.scale * 3)
hmeter2 = Unit.create((hmeter ** 2).dim, name="hmeter2", dispname=f"{str(hmeter)}^2", scale=hmeter.scale * 2)
hmeter3 = Unit.create((hmeter ** 3).dim, name="hmeter3", dispname=f"{str(hmeter)}^3", scale=hmeter.scale * 3)
dameter2 = Unit.create((dameter ** 2).dim, name="dameter2", dispname=f"{str(dameter)}^2", scale=dameter.scale * 2)
dameter3 = Unit.create((dameter ** 3).dim, name="dameter3", dispname=f"{str(dameter)}^3", scale=dameter.scale * 3)
mmeter2 = Unit.create((mmeter ** 2).dim, name="mmeter2", dispname=f"{str(mmeter)}^2", scale=mmeter.scale * 2)
mmeter3 = Unit.create((mmeter ** 3).dim, name="mmeter3", dispname=f"{str(mmeter)}^3", scale=mmeter.scale * 3)
nmeter2 = Unit.create((nmeter ** 2).dim, name="nmeter2", dispname=f"{str(nmeter)}^2", scale=nmeter.scale * 2)
nmeter3 = Unit.create((nmeter ** 3).dim, name="nmeter3", dispname=f"{str(nmeter)}^3", scale=nmeter.scale * 3)
pmeter2 = Unit.create((pmeter ** 2).dim, name="pmeter2", dispname=f"{str(pmeter)}^2", scale=pmeter.scale * 2)
pmeter3 = Unit.create((pmeter ** 3).dim, name="pmeter3", dispname=f"{str(pmeter)}^3", scale=pmeter.scale * 3)
umeter2 = Unit.create((umeter ** 2).dim, name="umeter2", dispname=f"{str(umeter)}^2", scale=umeter.scale * 2)
umeter3 = Unit.create((umeter ** 3).dim, name="umeter3", dispname=f"{str(umeter)}^3", scale=umeter.scale * 3)
Tmeter2 = Unit.create((Tmeter ** 2).dim, name="Tmeter2", dispname=f"{str(Tmeter)}^2", scale=Tmeter.scale * 2)
Tmeter3 = Unit.create((Tmeter ** 3).dim, name="Tmeter3", dispname=f"{str(Tmeter)}^3", scale=Tmeter.scale * 3)
ymeter2 = Unit.create((ymeter ** 2).dim, name="ymeter2", dispname=f"{str(ymeter)}^2", scale=ymeter.scale * 2)
ymeter3 = Unit.create((ymeter ** 3).dim, name="ymeter3", dispname=f"{str(ymeter)}^3", scale=ymeter.scale * 3)
Emeter2 = Unit.create((Emeter ** 2).dim, name="Emeter2", dispname=f"{str(Emeter)}^2", scale=Emeter.scale * 2)
Emeter3 = Unit.create((Emeter ** 3).dim, name="Emeter3", dispname=f"{str(Emeter)}^3", scale=Emeter.scale * 3)
zmeter2 = Unit.create((zmeter ** 2).dim, name="zmeter2", dispname=f"{str(zmeter)}^2", scale=zmeter.scale * 2)
zmeter3 = Unit.create((zmeter ** 3).dim, name="zmeter3", dispname=f"{str(zmeter)}^3", scale=zmeter.scale * 3)
Mmeter2 = Unit.create((Mmeter ** 2).dim, name="Mmeter2", dispname=f"{str(Mmeter)}^2", scale=Mmeter.scale * 2)
Mmeter3 = Unit.create((Mmeter ** 3).dim, name="Mmeter3", dispname=f"{str(Mmeter)}^3", scale=Mmeter.scale * 3)
kmeter2 = Unit.create((kmeter ** 2).dim, name="kmeter2", dispname=f"{str(kmeter)}^2", scale=kmeter.scale * 2)
kmeter3 = Unit.create((kmeter ** 3).dim, name="kmeter3", dispname=f"{str(kmeter)}^3", scale=kmeter.scale * 3)
Ymeter2 = Unit.create((Ymeter ** 2).dim, name="Ymeter2", dispname=f"{str(Ymeter)}^2", scale=Ymeter.scale * 2)
Ymeter3 = Unit.create((Ymeter ** 3).dim, name="Ymeter3", dispname=f"{str(Ymeter)}^3", scale=Ymeter.scale * 3)
asecond2 = Unit.create((asecond ** 2).dim, name="asecond2", dispname=f"{str(asecond)}^2", scale=asecond.scale * 2)
asecond3 = Unit.create((asecond ** 3).dim, name="asecond3", dispname=f"{str(asecond)}^3", scale=asecond.scale * 3)
csecond2 = Unit.create((csecond ** 2).dim, name="csecond2", dispname=f"{str(csecond)}^2", scale=csecond.scale * 2)
csecond3 = Unit.create((csecond ** 3).dim, name="csecond3", dispname=f"{str(csecond)}^3", scale=csecond.scale * 3)
Zsecond2 = Unit.create((Zsecond ** 2).dim, name="Zsecond2", dispname=f"{str(Zsecond)}^2", scale=Zsecond.scale * 2)
Zsecond3 = Unit.create((Zsecond ** 3).dim, name="Zsecond3", dispname=f"{str(Zsecond)}^3", scale=Zsecond.scale * 3)
Psecond2 = Unit.create((Psecond ** 2).dim, name="Psecond2", dispname=f"{str(Psecond)}^2", scale=Psecond.scale * 2)
Psecond3 = Unit.create((Psecond ** 3).dim, name="Psecond3", dispname=f"{str(Psecond)}^3", scale=Psecond.scale * 3)
dsecond2 = Unit.create((dsecond ** 2).dim, name="dsecond2", dispname=f"{str(dsecond)}^2", scale=dsecond.scale * 2)
dsecond3 = Unit.create((dsecond ** 3).dim, name="dsecond3", dispname=f"{str(dsecond)}^3", scale=dsecond.scale * 3)
Gsecond2 = Unit.create((Gsecond ** 2).dim, name="Gsecond2", dispname=f"{str(Gsecond)}^2", scale=Gsecond.scale * 2)
Gsecond3 = Unit.create((Gsecond ** 3).dim, name="Gsecond3", dispname=f"{str(Gsecond)}^3", scale=Gsecond.scale * 3)
fsecond2 = Unit.create((fsecond ** 2).dim, name="fsecond2", dispname=f"{str(fsecond)}^2", scale=fsecond.scale * 2)
fsecond3 = Unit.create((fsecond ** 3).dim, name="fsecond3", dispname=f"{str(fsecond)}^3", scale=fsecond.scale * 3)
hsecond2 = Unit.create((hsecond ** 2).dim, name="hsecond2", dispname=f"{str(hsecond)}^2", scale=hsecond.scale * 2)
hsecond3 = Unit.create((hsecond ** 3).dim, name="hsecond3", dispname=f"{str(hsecond)}^3", scale=hsecond.scale * 3)
dasecond2 = Unit.create((dasecond ** 2).dim, name="dasecond2", dispname=f"{str(dasecond)}^2",
                        scale=dasecond.scale * 2)
dasecond3 = Unit.create((dasecond ** 3).dim, name="dasecond3", dispname=f"{str(dasecond)}^3",
                        scale=dasecond.scale * 3)
msecond2 = Unit.create((msecond ** 2).dim, name="msecond2", dispname=f"{str(msecond)}^2", scale=msecond.scale * 2)
msecond3 = Unit.create((msecond ** 3).dim, name="msecond3", dispname=f"{str(msecond)}^3", scale=msecond.scale * 3)
nsecond2 = Unit.create((nsecond ** 2).dim, name="nsecond2", dispname=f"{str(nsecond)}^2", scale=nsecond.scale * 2)
nsecond3 = Unit.create((nsecond ** 3).dim, name="nsecond3", dispname=f"{str(nsecond)}^3", scale=nsecond.scale * 3)
psecond2 = Unit.create((psecond ** 2).dim, name="psecond2", dispname=f"{str(psecond)}^2", scale=psecond.scale * 2)
psecond3 = Unit.create((psecond ** 3).dim, name="psecond3", dispname=f"{str(psecond)}^3", scale=psecond.scale * 3)
usecond2 = Unit.create((usecond ** 2).dim, name="usecond2", dispname=f"{str(usecond)}^2", scale=usecond.scale * 2)
usecond3 = Unit.create((usecond ** 3).dim, name="usecond3", dispname=f"{str(usecond)}^3", scale=usecond.scale * 3)
Tsecond2 = Unit.create((Tsecond ** 2).dim, name="Tsecond2", dispname=f"{str(Tsecond)}^2", scale=Tsecond.scale * 2)
Tsecond3 = Unit.create((Tsecond ** 3).dim, name="Tsecond3", dispname=f"{str(Tsecond)}^3", scale=Tsecond.scale * 3)
ysecond2 = Unit.create((ysecond ** 2).dim, name="ysecond2", dispname=f"{str(ysecond)}^2", scale=ysecond.scale * 2)
ysecond3 = Unit.create((ysecond ** 3).dim, name="ysecond3", dispname=f"{str(ysecond)}^3", scale=ysecond.scale * 3)
Esecond2 = Unit.create((Esecond ** 2).dim, name="Esecond2", dispname=f"{str(Esecond)}^2", scale=Esecond.scale * 2)
Esecond3 = Unit.create((Esecond ** 3).dim, name="Esecond3", dispname=f"{str(Esecond)}^3", scale=Esecond.scale * 3)
zsecond2 = Unit.create((zsecond ** 2).dim, name="zsecond2", dispname=f"{str(zsecond)}^2", scale=zsecond.scale * 2)
zsecond3 = Unit.create((zsecond ** 3).dim, name="zsecond3", dispname=f"{str(zsecond)}^3", scale=zsecond.scale * 3)
Msecond2 = Unit.create((Msecond ** 2).dim, name="Msecond2", dispname=f"{str(Msecond)}^2", scale=Msecond.scale * 2)
Msecond3 = Unit.create((Msecond ** 3).dim, name="Msecond3", dispname=f"{str(Msecond)}^3", scale=Msecond.scale * 3)
ksecond2 = Unit.create((ksecond ** 2).dim, name="ksecond2", dispname=f"{str(ksecond)}^2", scale=ksecond.scale * 2)
ksecond3 = Unit.create((ksecond ** 3).dim, name="ksecond3", dispname=f"{str(ksecond)}^3", scale=ksecond.scale * 3)
Ysecond2 = Unit.create((Ysecond ** 2).dim, name="Ysecond2", dispname=f"{str(Ysecond)}^2", scale=Ysecond.scale * 2)
Ysecond3 = Unit.create((Ysecond ** 3).dim, name="Ysecond3", dispname=f"{str(Ysecond)}^3", scale=Ysecond.scale * 3)
aamp2 = Unit.create((aamp ** 2).dim, name="aamp2", dispname=f"{str(aamp)}^2", scale=aamp.scale * 2)
aamp3 = Unit.create((aamp ** 3).dim, name="aamp3", dispname=f"{str(aamp)}^3", scale=aamp.scale * 3)
camp2 = Unit.create((camp ** 2).dim, name="camp2", dispname=f"{str(camp)}^2", scale=camp.scale * 2)
camp3 = Unit.create((camp ** 3).dim, name="camp3", dispname=f"{str(camp)}^3", scale=camp.scale * 3)
Zamp2 = Unit.create((Zamp ** 2).dim, name="Zamp2", dispname=f"{str(Zamp)}^2", scale=Zamp.scale * 2)
Zamp3 = Unit.create((Zamp ** 3).dim, name="Zamp3", dispname=f"{str(Zamp)}^3", scale=Zamp.scale * 3)
Pamp2 = Unit.create((Pamp ** 2).dim, name="Pamp2", dispname=f"{str(Pamp)}^2", scale=Pamp.scale * 2)
Pamp3 = Unit.create((Pamp ** 3).dim, name="Pamp3", dispname=f"{str(Pamp)}^3", scale=Pamp.scale * 3)
damp2 = Unit.create((damp ** 2).dim, name="damp2", dispname=f"{str(damp)}^2", scale=damp.scale * 2)
damp3 = Unit.create((damp ** 3).dim, name="damp3", dispname=f"{str(damp)}^3", scale=damp.scale * 3)
Gamp2 = Unit.create((Gamp ** 2).dim, name="Gamp2", dispname=f"{str(Gamp)}^2", scale=Gamp.scale * 2)
Gamp3 = Unit.create((Gamp ** 3).dim, name="Gamp3", dispname=f"{str(Gamp)}^3", scale=Gamp.scale * 3)
famp2 = Unit.create((famp ** 2).dim, name="famp2", dispname=f"{str(famp)}^2", scale=famp.scale * 2)
famp3 = Unit.create((famp ** 3).dim, name="famp3", dispname=f"{str(famp)}^3", scale=famp.scale * 3)
hamp2 = Unit.create((hamp ** 2).dim, name="hamp2", dispname=f"{str(hamp)}^2", scale=hamp.scale * 2)
hamp3 = Unit.create((hamp ** 3).dim, name="hamp3", dispname=f"{str(hamp)}^3", scale=hamp.scale * 3)
daamp2 = Unit.create((daamp ** 2).dim, name="daamp2", dispname=f"{str(daamp)}^2", scale=daamp.scale * 2)
daamp3 = Unit.create((daamp ** 3).dim, name="daamp3", dispname=f"{str(daamp)}^3", scale=daamp.scale * 3)
mamp2 = Unit.create((mamp ** 2).dim, name="mamp2", dispname=f"{str(mamp)}^2", scale=mamp.scale * 2)
mamp3 = Unit.create((mamp ** 3).dim, name="mamp3", dispname=f"{str(mamp)}^3", scale=mamp.scale * 3)
namp2 = Unit.create((namp ** 2).dim, name="namp2", dispname=f"{str(namp)}^2", scale=namp.scale * 2)
namp3 = Unit.create((namp ** 3).dim, name="namp3", dispname=f"{str(namp)}^3", scale=namp.scale * 3)
pamp2 = Unit.create((pamp ** 2).dim, name="pamp2", dispname=f"{str(pamp)}^2", scale=pamp.scale * 2)
pamp3 = Unit.create((pamp ** 3).dim, name="pamp3", dispname=f"{str(pamp)}^3", scale=pamp.scale * 3)
uamp2 = Unit.create((uamp ** 2).dim, name="uamp2", dispname=f"{str(uamp)}^2", scale=uamp.scale * 2)
uamp3 = Unit.create((uamp ** 3).dim, name="uamp3", dispname=f"{str(uamp)}^3", scale=uamp.scale * 3)
Tamp2 = Unit.create((Tamp ** 2).dim, name="Tamp2", dispname=f"{str(Tamp)}^2", scale=Tamp.scale * 2)
Tamp3 = Unit.create((Tamp ** 3).dim, name="Tamp3", dispname=f"{str(Tamp)}^3", scale=Tamp.scale * 3)
yamp2 = Unit.create((yamp ** 2).dim, name="yamp2", dispname=f"{str(yamp)}^2", scale=yamp.scale * 2)
yamp3 = Unit.create((yamp ** 3).dim, name="yamp3", dispname=f"{str(yamp)}^3", scale=yamp.scale * 3)
Eamp2 = Unit.create((Eamp ** 2).dim, name="Eamp2", dispname=f"{str(Eamp)}^2", scale=Eamp.scale * 2)
Eamp3 = Unit.create((Eamp ** 3).dim, name="Eamp3", dispname=f"{str(Eamp)}^3", scale=Eamp.scale * 3)
zamp2 = Unit.create((zamp ** 2).dim, name="zamp2", dispname=f"{str(zamp)}^2", scale=zamp.scale * 2)
zamp3 = Unit.create((zamp ** 3).dim, name="zamp3", dispname=f"{str(zamp)}^3", scale=zamp.scale * 3)
Mamp2 = Unit.create((Mamp ** 2).dim, name="Mamp2", dispname=f"{str(Mamp)}^2", scale=Mamp.scale * 2)
Mamp3 = Unit.create((Mamp ** 3).dim, name="Mamp3", dispname=f"{str(Mamp)}^3", scale=Mamp.scale * 3)
kamp2 = Unit.create((kamp ** 2).dim, name="kamp2", dispname=f"{str(kamp)}^2", scale=kamp.scale * 2)
kamp3 = Unit.create((kamp ** 3).dim, name="kamp3", dispname=f"{str(kamp)}^3", scale=kamp.scale * 3)
Yamp2 = Unit.create((Yamp ** 2).dim, name="Yamp2", dispname=f"{str(Yamp)}^2", scale=Yamp.scale * 2)
Yamp3 = Unit.create((Yamp ** 3).dim, name="Yamp3", dispname=f"{str(Yamp)}^3", scale=Yamp.scale * 3)
aampere2 = Unit.create((aampere ** 2).dim, name="aampere2", dispname=f"{str(aampere)}^2", scale=aampere.scale * 2)
aampere3 = Unit.create((aampere ** 3).dim, name="aampere3", dispname=f"{str(aampere)}^3", scale=aampere.scale * 3)
campere2 = Unit.create((campere ** 2).dim, name="campere2", dispname=f"{str(campere)}^2", scale=campere.scale * 2)
campere3 = Unit.create((campere ** 3).dim, name="campere3", dispname=f"{str(campere)}^3", scale=campere.scale * 3)
Zampere2 = Unit.create((Zampere ** 2).dim, name="Zampere2", dispname=f"{str(Zampere)}^2", scale=Zampere.scale * 2)
Zampere3 = Unit.create((Zampere ** 3).dim, name="Zampere3", dispname=f"{str(Zampere)}^3", scale=Zampere.scale * 3)
Pampere2 = Unit.create((Pampere ** 2).dim, name="Pampere2", dispname=f"{str(Pampere)}^2", scale=Pampere.scale * 2)
Pampere3 = Unit.create((Pampere ** 3).dim, name="Pampere3", dispname=f"{str(Pampere)}^3", scale=Pampere.scale * 3)
dampere2 = Unit.create((dampere ** 2).dim, name="dampere2", dispname=f"{str(dampere)}^2", scale=dampere.scale * 2)
dampere3 = Unit.create((dampere ** 3).dim, name="dampere3", dispname=f"{str(dampere)}^3", scale=dampere.scale * 3)
Gampere2 = Unit.create((Gampere ** 2).dim, name="Gampere2", dispname=f"{str(Gampere)}^2", scale=Gampere.scale * 2)
Gampere3 = Unit.create((Gampere ** 3).dim, name="Gampere3", dispname=f"{str(Gampere)}^3", scale=Gampere.scale * 3)
fampere2 = Unit.create((fampere ** 2).dim, name="fampere2", dispname=f"{str(fampere)}^2", scale=fampere.scale * 2)
fampere3 = Unit.create((fampere ** 3).dim, name="fampere3", dispname=f"{str(fampere)}^3", scale=fampere.scale * 3)
hampere2 = Unit.create((hampere ** 2).dim, name="hampere2", dispname=f"{str(hampere)}^2", scale=hampere.scale * 2)
hampere3 = Unit.create((hampere ** 3).dim, name="hampere3", dispname=f"{str(hampere)}^3", scale=hampere.scale * 3)
daampere2 = Unit.create((daampere ** 2).dim, name="daampere2", dispname=f"{str(daampere)}^2",
                        scale=daampere.scale * 2)
daampere3 = Unit.create((daampere ** 3).dim, name="daampere3", dispname=f"{str(daampere)}^3",
                        scale=daampere.scale * 3)
mampere2 = Unit.create((mampere ** 2).dim, name="mampere2", dispname=f"{str(mampere)}^2", scale=mampere.scale * 2)
mampere3 = Unit.create((mampere ** 3).dim, name="mampere3", dispname=f"{str(mampere)}^3", scale=mampere.scale * 3)
nampere2 = Unit.create((nampere ** 2).dim, name="nampere2", dispname=f"{str(nampere)}^2", scale=nampere.scale * 2)
nampere3 = Unit.create((nampere ** 3).dim, name="nampere3", dispname=f"{str(nampere)}^3", scale=nampere.scale * 3)
pampere2 = Unit.create((pampere ** 2).dim, name="pampere2", dispname=f"{str(pampere)}^2", scale=pampere.scale * 2)
pampere3 = Unit.create((pampere ** 3).dim, name="pampere3", dispname=f"{str(pampere)}^3", scale=pampere.scale * 3)
uampere2 = Unit.create((uampere ** 2).dim, name="uampere2", dispname=f"{str(uampere)}^2", scale=uampere.scale * 2)
uampere3 = Unit.create((uampere ** 3).dim, name="uampere3", dispname=f"{str(uampere)}^3", scale=uampere.scale * 3)
Tampere2 = Unit.create((Tampere ** 2).dim, name="Tampere2", dispname=f"{str(Tampere)}^2", scale=Tampere.scale * 2)
Tampere3 = Unit.create((Tampere ** 3).dim, name="Tampere3", dispname=f"{str(Tampere)}^3", scale=Tampere.scale * 3)
yampere2 = Unit.create((yampere ** 2).dim, name="yampere2", dispname=f"{str(yampere)}^2", scale=yampere.scale * 2)
yampere3 = Unit.create((yampere ** 3).dim, name="yampere3", dispname=f"{str(yampere)}^3", scale=yampere.scale * 3)
Eampere2 = Unit.create((Eampere ** 2).dim, name="Eampere2", dispname=f"{str(Eampere)}^2", scale=Eampere.scale * 2)
Eampere3 = Unit.create((Eampere ** 3).dim, name="Eampere3", dispname=f"{str(Eampere)}^3", scale=Eampere.scale * 3)
zampere2 = Unit.create((zampere ** 2).dim, name="zampere2", dispname=f"{str(zampere)}^2", scale=zampere.scale * 2)
zampere3 = Unit.create((zampere ** 3).dim, name="zampere3", dispname=f"{str(zampere)}^3", scale=zampere.scale * 3)
Mampere2 = Unit.create((Mampere ** 2).dim, name="Mampere2", dispname=f"{str(Mampere)}^2", scale=Mampere.scale * 2)
Mampere3 = Unit.create((Mampere ** 3).dim, name="Mampere3", dispname=f"{str(Mampere)}^3", scale=Mampere.scale * 3)
kampere2 = Unit.create((kampere ** 2).dim, name="kampere2", dispname=f"{str(kampere)}^2", scale=kampere.scale * 2)
kampere3 = Unit.create((kampere ** 3).dim, name="kampere3", dispname=f"{str(kampere)}^3", scale=kampere.scale * 3)
Yampere2 = Unit.create((Yampere ** 2).dim, name="Yampere2", dispname=f"{str(Yampere)}^2", scale=Yampere.scale * 2)
Yampere3 = Unit.create((Yampere ** 3).dim, name="Yampere3", dispname=f"{str(Yampere)}^3", scale=Yampere.scale * 3)
amole2 = Unit.create((amole ** 2).dim, name="amole2", dispname=f"{str(amole)}^2", scale=amole.scale * 2)
amole3 = Unit.create((amole ** 3).dim, name="amole3", dispname=f"{str(amole)}^3", scale=amole.scale * 3)
cmole2 = Unit.create((cmole ** 2).dim, name="cmole2", dispname=f"{str(cmole)}^2", scale=cmole.scale * 2)
cmole3 = Unit.create((cmole ** 3).dim, name="cmole3", dispname=f"{str(cmole)}^3", scale=cmole.scale * 3)
Zmole2 = Unit.create((Zmole ** 2).dim, name="Zmole2", dispname=f"{str(Zmole)}^2", scale=Zmole.scale * 2)
Zmole3 = Unit.create((Zmole ** 3).dim, name="Zmole3", dispname=f"{str(Zmole)}^3", scale=Zmole.scale * 3)
Pmole2 = Unit.create((Pmole ** 2).dim, name="Pmole2", dispname=f"{str(Pmole)}^2", scale=Pmole.scale * 2)
Pmole3 = Unit.create((Pmole ** 3).dim, name="Pmole3", dispname=f"{str(Pmole)}^3", scale=Pmole.scale * 3)
dmole2 = Unit.create((dmole ** 2).dim, name="dmole2", dispname=f"{str(dmole)}^2", scale=dmole.scale * 2)
dmole3 = Unit.create((dmole ** 3).dim, name="dmole3", dispname=f"{str(dmole)}^3", scale=dmole.scale * 3)
Gmole2 = Unit.create((Gmole ** 2).dim, name="Gmole2", dispname=f"{str(Gmole)}^2", scale=Gmole.scale * 2)
Gmole3 = Unit.create((Gmole ** 3).dim, name="Gmole3", dispname=f"{str(Gmole)}^3", scale=Gmole.scale * 3)
fmole2 = Unit.create((fmole ** 2).dim, name="fmole2", dispname=f"{str(fmole)}^2", scale=fmole.scale * 2)
fmole3 = Unit.create((fmole ** 3).dim, name="fmole3", dispname=f"{str(fmole)}^3", scale=fmole.scale * 3)
hmole2 = Unit.create((hmole ** 2).dim, name="hmole2", dispname=f"{str(hmole)}^2", scale=hmole.scale * 2)
hmole3 = Unit.create((hmole ** 3).dim, name="hmole3", dispname=f"{str(hmole)}^3", scale=hmole.scale * 3)
damole2 = Unit.create((damole ** 2).dim, name="damole2", dispname=f"{str(damole)}^2", scale=damole.scale * 2)
damole3 = Unit.create((damole ** 3).dim, name="damole3", dispname=f"{str(damole)}^3", scale=damole.scale * 3)
mmole2 = Unit.create((mmole ** 2).dim, name="mmole2", dispname=f"{str(mmole)}^2", scale=mmole.scale * 2)
mmole3 = Unit.create((mmole ** 3).dim, name="mmole3", dispname=f"{str(mmole)}^3", scale=mmole.scale * 3)
nmole2 = Unit.create((nmole ** 2).dim, name="nmole2", dispname=f"{str(nmole)}^2", scale=nmole.scale * 2)
nmole3 = Unit.create((nmole ** 3).dim, name="nmole3", dispname=f"{str(nmole)}^3", scale=nmole.scale * 3)
pmole2 = Unit.create((pmole ** 2).dim, name="pmole2", dispname=f"{str(pmole)}^2", scale=pmole.scale * 2)
pmole3 = Unit.create((pmole ** 3).dim, name="pmole3", dispname=f"{str(pmole)}^3", scale=pmole.scale * 3)
umole2 = Unit.create((umole ** 2).dim, name="umole2", dispname=f"{str(umole)}^2", scale=umole.scale * 2)
umole3 = Unit.create((umole ** 3).dim, name="umole3", dispname=f"{str(umole)}^3", scale=umole.scale * 3)
Tmole2 = Unit.create((Tmole ** 2).dim, name="Tmole2", dispname=f"{str(Tmole)}^2", scale=Tmole.scale * 2)
Tmole3 = Unit.create((Tmole ** 3).dim, name="Tmole3", dispname=f"{str(Tmole)}^3", scale=Tmole.scale * 3)
ymole2 = Unit.create((ymole ** 2).dim, name="ymole2", dispname=f"{str(ymole)}^2", scale=ymole.scale * 2)
ymole3 = Unit.create((ymole ** 3).dim, name="ymole3", dispname=f"{str(ymole)}^3", scale=ymole.scale * 3)
Emole2 = Unit.create((Emole ** 2).dim, name="Emole2", dispname=f"{str(Emole)}^2", scale=Emole.scale * 2)
Emole3 = Unit.create((Emole ** 3).dim, name="Emole3", dispname=f"{str(Emole)}^3", scale=Emole.scale * 3)
zmole2 = Unit.create((zmole ** 2).dim, name="zmole2", dispname=f"{str(zmole)}^2", scale=zmole.scale * 2)
zmole3 = Unit.create((zmole ** 3).dim, name="zmole3", dispname=f"{str(zmole)}^3", scale=zmole.scale * 3)
Mmole2 = Unit.create((Mmole ** 2).dim, name="Mmole2", dispname=f"{str(Mmole)}^2", scale=Mmole.scale * 2)
Mmole3 = Unit.create((Mmole ** 3).dim, name="Mmole3", dispname=f"{str(Mmole)}^3", scale=Mmole.scale * 3)
kmole2 = Unit.create((kmole ** 2).dim, name="kmole2", dispname=f"{str(kmole)}^2", scale=kmole.scale * 2)
kmole3 = Unit.create((kmole ** 3).dim, name="kmole3", dispname=f"{str(kmole)}^3", scale=kmole.scale * 3)
Ymole2 = Unit.create((Ymole ** 2).dim, name="Ymole2", dispname=f"{str(Ymole)}^2", scale=Ymole.scale * 2)
Ymole3 = Unit.create((Ymole ** 3).dim, name="Ymole3", dispname=f"{str(Ymole)}^3", scale=Ymole.scale * 3)
amol2 = Unit.create((amol ** 2).dim, name="amol2", dispname=f"{str(amol)}^2", scale=amol.scale * 2)
amol3 = Unit.create((amol ** 3).dim, name="amol3", dispname=f"{str(amol)}^3", scale=amol.scale * 3)
cmol2 = Unit.create((cmol ** 2).dim, name="cmol2", dispname=f"{str(cmol)}^2", scale=cmol.scale * 2)
cmol3 = Unit.create((cmol ** 3).dim, name="cmol3", dispname=f"{str(cmol)}^3", scale=cmol.scale * 3)
Zmol2 = Unit.create((Zmol ** 2).dim, name="Zmol2", dispname=f"{str(Zmol)}^2", scale=Zmol.scale * 2)
Zmol3 = Unit.create((Zmol ** 3).dim, name="Zmol3", dispname=f"{str(Zmol)}^3", scale=Zmol.scale * 3)
Pmol2 = Unit.create((Pmol ** 2).dim, name="Pmol2", dispname=f"{str(Pmol)}^2", scale=Pmol.scale * 2)
Pmol3 = Unit.create((Pmol ** 3).dim, name="Pmol3", dispname=f"{str(Pmol)}^3", scale=Pmol.scale * 3)
dmol2 = Unit.create((dmol ** 2).dim, name="dmol2", dispname=f"{str(dmol)}^2", scale=dmol.scale * 2)
dmol3 = Unit.create((dmol ** 3).dim, name="dmol3", dispname=f"{str(dmol)}^3", scale=dmol.scale * 3)
Gmol2 = Unit.create((Gmol ** 2).dim, name="Gmol2", dispname=f"{str(Gmol)}^2", scale=Gmol.scale * 2)
Gmol3 = Unit.create((Gmol ** 3).dim, name="Gmol3", dispname=f"{str(Gmol)}^3", scale=Gmol.scale * 3)
fmol2 = Unit.create((fmol ** 2).dim, name="fmol2", dispname=f"{str(fmol)}^2", scale=fmol.scale * 2)
fmol3 = Unit.create((fmol ** 3).dim, name="fmol3", dispname=f"{str(fmol)}^3", scale=fmol.scale * 3)
hmol2 = Unit.create((hmol ** 2).dim, name="hmol2", dispname=f"{str(hmol)}^2", scale=hmol.scale * 2)
hmol3 = Unit.create((hmol ** 3).dim, name="hmol3", dispname=f"{str(hmol)}^3", scale=hmol.scale * 3)
damol2 = Unit.create((damol ** 2).dim, name="damol2", dispname=f"{str(damol)}^2", scale=damol.scale * 2)
damol3 = Unit.create((damol ** 3).dim, name="damol3", dispname=f"{str(damol)}^3", scale=damol.scale * 3)
mmol2 = Unit.create((mmol ** 2).dim, name="mmol2", dispname=f"{str(mmol)}^2", scale=mmol.scale * 2)
mmol3 = Unit.create((mmol ** 3).dim, name="mmol3", dispname=f"{str(mmol)}^3", scale=mmol.scale * 3)
nmol2 = Unit.create((nmol ** 2).dim, name="nmol2", dispname=f"{str(nmol)}^2", scale=nmol.scale * 2)
nmol3 = Unit.create((nmol ** 3).dim, name="nmol3", dispname=f"{str(nmol)}^3", scale=nmol.scale * 3)
pmol2 = Unit.create((pmol ** 2).dim, name="pmol2", dispname=f"{str(pmol)}^2", scale=pmol.scale * 2)
pmol3 = Unit.create((pmol ** 3).dim, name="pmol3", dispname=f"{str(pmol)}^3", scale=pmol.scale * 3)
umol2 = Unit.create((umol ** 2).dim, name="umol2", dispname=f"{str(umol)}^2", scale=umol.scale * 2)
umol3 = Unit.create((umol ** 3).dim, name="umol3", dispname=f"{str(umol)}^3", scale=umol.scale * 3)
Tmol2 = Unit.create((Tmol ** 2).dim, name="Tmol2", dispname=f"{str(Tmol)}^2", scale=Tmol.scale * 2)
Tmol3 = Unit.create((Tmol ** 3).dim, name="Tmol3", dispname=f"{str(Tmol)}^3", scale=Tmol.scale * 3)
ymol2 = Unit.create((ymol ** 2).dim, name="ymol2", dispname=f"{str(ymol)}^2", scale=ymol.scale * 2)
ymol3 = Unit.create((ymol ** 3).dim, name="ymol3", dispname=f"{str(ymol)}^3", scale=ymol.scale * 3)
Emol2 = Unit.create((Emol ** 2).dim, name="Emol2", dispname=f"{str(Emol)}^2", scale=Emol.scale * 2)
Emol3 = Unit.create((Emol ** 3).dim, name="Emol3", dispname=f"{str(Emol)}^3", scale=Emol.scale * 3)
zmol2 = Unit.create((zmol ** 2).dim, name="zmol2", dispname=f"{str(zmol)}^2", scale=zmol.scale * 2)
zmol3 = Unit.create((zmol ** 3).dim, name="zmol3", dispname=f"{str(zmol)}^3", scale=zmol.scale * 3)
Mmol2 = Unit.create((Mmol ** 2).dim, name="Mmol2", dispname=f"{str(Mmol)}^2", scale=Mmol.scale * 2)
Mmol3 = Unit.create((Mmol ** 3).dim, name="Mmol3", dispname=f"{str(Mmol)}^3", scale=Mmol.scale * 3)
kmol2 = Unit.create((kmol ** 2).dim, name="kmol2", dispname=f"{str(kmol)}^2", scale=kmol.scale * 2)
kmol3 = Unit.create((kmol ** 3).dim, name="kmol3", dispname=f"{str(kmol)}^3", scale=kmol.scale * 3)
Ymol2 = Unit.create((Ymol ** 2).dim, name="Ymol2", dispname=f"{str(Ymol)}^2", scale=Ymol.scale * 2)
Ymol3 = Unit.create((Ymol ** 3).dim, name="Ymol3", dispname=f"{str(Ymol)}^3", scale=Ymol.scale * 3)
acandle2 = Unit.create((acandle ** 2).dim, name="acandle2", dispname=f"{str(acandle)}^2", scale=acandle.scale * 2)
acandle3 = Unit.create((acandle ** 3).dim, name="acandle3", dispname=f"{str(acandle)}^3", scale=acandle.scale * 3)
ccandle2 = Unit.create((ccandle ** 2).dim, name="ccandle2", dispname=f"{str(ccandle)}^2", scale=ccandle.scale * 2)
ccandle3 = Unit.create((ccandle ** 3).dim, name="ccandle3", dispname=f"{str(ccandle)}^3", scale=ccandle.scale * 3)
Zcandle2 = Unit.create((Zcandle ** 2).dim, name="Zcandle2", dispname=f"{str(Zcandle)}^2", scale=Zcandle.scale * 2)
Zcandle3 = Unit.create((Zcandle ** 3).dim, name="Zcandle3", dispname=f"{str(Zcandle)}^3", scale=Zcandle.scale * 3)
Pcandle2 = Unit.create((Pcandle ** 2).dim, name="Pcandle2", dispname=f"{str(Pcandle)}^2", scale=Pcandle.scale * 2)
Pcandle3 = Unit.create((Pcandle ** 3).dim, name="Pcandle3", dispname=f"{str(Pcandle)}^3", scale=Pcandle.scale * 3)
dcandle2 = Unit.create((dcandle ** 2).dim, name="dcandle2", dispname=f"{str(dcandle)}^2", scale=dcandle.scale * 2)
dcandle3 = Unit.create((dcandle ** 3).dim, name="dcandle3", dispname=f"{str(dcandle)}^3", scale=dcandle.scale * 3)
Gcandle2 = Unit.create((Gcandle ** 2).dim, name="Gcandle2", dispname=f"{str(Gcandle)}^2", scale=Gcandle.scale * 2)
Gcandle3 = Unit.create((Gcandle ** 3).dim, name="Gcandle3", dispname=f"{str(Gcandle)}^3", scale=Gcandle.scale * 3)
fcandle2 = Unit.create((fcandle ** 2).dim, name="fcandle2", dispname=f"{str(fcandle)}^2", scale=fcandle.scale * 2)
fcandle3 = Unit.create((fcandle ** 3).dim, name="fcandle3", dispname=f"{str(fcandle)}^3", scale=fcandle.scale * 3)
hcandle2 = Unit.create((hcandle ** 2).dim, name="hcandle2", dispname=f"{str(hcandle)}^2", scale=hcandle.scale * 2)
hcandle3 = Unit.create((hcandle ** 3).dim, name="hcandle3", dispname=f"{str(hcandle)}^3", scale=hcandle.scale * 3)
dacandle2 = Unit.create((dacandle ** 2).dim, name="dacandle2", dispname=f"{str(dacandle)}^2",
                        scale=dacandle.scale * 2)
dacandle3 = Unit.create((dacandle ** 3).dim, name="dacandle3", dispname=f"{str(dacandle)}^3",
                        scale=dacandle.scale * 3)
mcandle2 = Unit.create((mcandle ** 2).dim, name="mcandle2", dispname=f"{str(mcandle)}^2", scale=mcandle.scale * 2)
mcandle3 = Unit.create((mcandle ** 3).dim, name="mcandle3", dispname=f"{str(mcandle)}^3", scale=mcandle.scale * 3)
ncandle2 = Unit.create((ncandle ** 2).dim, name="ncandle2", dispname=f"{str(ncandle)}^2", scale=ncandle.scale * 2)
ncandle3 = Unit.create((ncandle ** 3).dim, name="ncandle3", dispname=f"{str(ncandle)}^3", scale=ncandle.scale * 3)
pcandle2 = Unit.create((pcandle ** 2).dim, name="pcandle2", dispname=f"{str(pcandle)}^2", scale=pcandle.scale * 2)
pcandle3 = Unit.create((pcandle ** 3).dim, name="pcandle3", dispname=f"{str(pcandle)}^3", scale=pcandle.scale * 3)
ucandle2 = Unit.create((ucandle ** 2).dim, name="ucandle2", dispname=f"{str(ucandle)}^2", scale=ucandle.scale * 2)
ucandle3 = Unit.create((ucandle ** 3).dim, name="ucandle3", dispname=f"{str(ucandle)}^3", scale=ucandle.scale * 3)
Tcandle2 = Unit.create((Tcandle ** 2).dim, name="Tcandle2", dispname=f"{str(Tcandle)}^2", scale=Tcandle.scale * 2)
Tcandle3 = Unit.create((Tcandle ** 3).dim, name="Tcandle3", dispname=f"{str(Tcandle)}^3", scale=Tcandle.scale * 3)
ycandle2 = Unit.create((ycandle ** 2).dim, name="ycandle2", dispname=f"{str(ycandle)}^2", scale=ycandle.scale * 2)
ycandle3 = Unit.create((ycandle ** 3).dim, name="ycandle3", dispname=f"{str(ycandle)}^3", scale=ycandle.scale * 3)
Ecandle2 = Unit.create((Ecandle ** 2).dim, name="Ecandle2", dispname=f"{str(Ecandle)}^2", scale=Ecandle.scale * 2)
Ecandle3 = Unit.create((Ecandle ** 3).dim, name="Ecandle3", dispname=f"{str(Ecandle)}^3", scale=Ecandle.scale * 3)
zcandle2 = Unit.create((zcandle ** 2).dim, name="zcandle2", dispname=f"{str(zcandle)}^2", scale=zcandle.scale * 2)
zcandle3 = Unit.create((zcandle ** 3).dim, name="zcandle3", dispname=f"{str(zcandle)}^3", scale=zcandle.scale * 3)
Mcandle2 = Unit.create((Mcandle ** 2).dim, name="Mcandle2", dispname=f"{str(Mcandle)}^2", scale=Mcandle.scale * 2)
Mcandle3 = Unit.create((Mcandle ** 3).dim, name="Mcandle3", dispname=f"{str(Mcandle)}^3", scale=Mcandle.scale * 3)
kcandle2 = Unit.create((kcandle ** 2).dim, name="kcandle2", dispname=f"{str(kcandle)}^2", scale=kcandle.scale * 2)
kcandle3 = Unit.create((kcandle ** 3).dim, name="kcandle3", dispname=f"{str(kcandle)}^3", scale=kcandle.scale * 3)
Ycandle2 = Unit.create((Ycandle ** 2).dim, name="Ycandle2", dispname=f"{str(Ycandle)}^2", scale=Ycandle.scale * 2)
Ycandle3 = Unit.create((Ycandle ** 3).dim, name="Ycandle3", dispname=f"{str(Ycandle)}^3", scale=Ycandle.scale * 3)
agram2 = Unit.create((agram ** 2).dim, name="agram2", dispname=f"{str(agram)}^2", scale=agram.scale * 2)
agram3 = Unit.create((agram ** 3).dim, name="agram3", dispname=f"{str(agram)}^3", scale=agram.scale * 3)
cgram2 = Unit.create((cgram ** 2).dim, name="cgram2", dispname=f"{str(cgram)}^2", scale=cgram.scale * 2)
cgram3 = Unit.create((cgram ** 3).dim, name="cgram3", dispname=f"{str(cgram)}^3", scale=cgram.scale * 3)
Zgram2 = Unit.create((Zgram ** 2).dim, name="Zgram2", dispname=f"{str(Zgram)}^2", scale=Zgram.scale * 2)
Zgram3 = Unit.create((Zgram ** 3).dim, name="Zgram3", dispname=f"{str(Zgram)}^3", scale=Zgram.scale * 3)
Pgram2 = Unit.create((Pgram ** 2).dim, name="Pgram2", dispname=f"{str(Pgram)}^2", scale=Pgram.scale * 2)
Pgram3 = Unit.create((Pgram ** 3).dim, name="Pgram3", dispname=f"{str(Pgram)}^3", scale=Pgram.scale * 3)
dgram2 = Unit.create((dgram ** 2).dim, name="dgram2", dispname=f"{str(dgram)}^2", scale=dgram.scale * 2)
dgram3 = Unit.create((dgram ** 3).dim, name="dgram3", dispname=f"{str(dgram)}^3", scale=dgram.scale * 3)
Ggram2 = Unit.create((Ggram ** 2).dim, name="Ggram2", dispname=f"{str(Ggram)}^2", scale=Ggram.scale * 2)
Ggram3 = Unit.create((Ggram ** 3).dim, name="Ggram3", dispname=f"{str(Ggram)}^3", scale=Ggram.scale * 3)
fgram2 = Unit.create((fgram ** 2).dim, name="fgram2", dispname=f"{str(fgram)}^2", scale=fgram.scale * 2)
fgram3 = Unit.create((fgram ** 3).dim, name="fgram3", dispname=f"{str(fgram)}^3", scale=fgram.scale * 3)
hgram2 = Unit.create((hgram ** 2).dim, name="hgram2", dispname=f"{str(hgram)}^2", scale=hgram.scale * 2)
hgram3 = Unit.create((hgram ** 3).dim, name="hgram3", dispname=f"{str(hgram)}^3", scale=hgram.scale * 3)
dagram2 = Unit.create((dagram ** 2).dim, name="dagram2", dispname=f"{str(dagram)}^2", scale=dagram.scale * 2)
dagram3 = Unit.create((dagram ** 3).dim, name="dagram3", dispname=f"{str(dagram)}^3", scale=dagram.scale * 3)
mgram2 = Unit.create((mgram ** 2).dim, name="mgram2", dispname=f"{str(mgram)}^2", scale=mgram.scale * 2)
mgram3 = Unit.create((mgram ** 3).dim, name="mgram3", dispname=f"{str(mgram)}^3", scale=mgram.scale * 3)
ngram2 = Unit.create((ngram ** 2).dim, name="ngram2", dispname=f"{str(ngram)}^2", scale=ngram.scale * 2)
ngram3 = Unit.create((ngram ** 3).dim, name="ngram3", dispname=f"{str(ngram)}^3", scale=ngram.scale * 3)
pgram2 = Unit.create((pgram ** 2).dim, name="pgram2", dispname=f"{str(pgram)}^2", scale=pgram.scale * 2)
pgram3 = Unit.create((pgram ** 3).dim, name="pgram3", dispname=f"{str(pgram)}^3", scale=pgram.scale * 3)
ugram2 = Unit.create((ugram ** 2).dim, name="ugram2", dispname=f"{str(ugram)}^2", scale=ugram.scale * 2)
ugram3 = Unit.create((ugram ** 3).dim, name="ugram3", dispname=f"{str(ugram)}^3", scale=ugram.scale * 3)
Tgram2 = Unit.create((Tgram ** 2).dim, name="Tgram2", dispname=f"{str(Tgram)}^2", scale=Tgram.scale * 2)
Tgram3 = Unit.create((Tgram ** 3).dim, name="Tgram3", dispname=f"{str(Tgram)}^3", scale=Tgram.scale * 3)
ygram2 = Unit.create((ygram ** 2).dim, name="ygram2", dispname=f"{str(ygram)}^2", scale=ygram.scale * 2)
ygram3 = Unit.create((ygram ** 3).dim, name="ygram3", dispname=f"{str(ygram)}^3", scale=ygram.scale * 3)
Egram2 = Unit.create((Egram ** 2).dim, name="Egram2", dispname=f"{str(Egram)}^2", scale=Egram.scale * 2)
Egram3 = Unit.create((Egram ** 3).dim, name="Egram3", dispname=f"{str(Egram)}^3", scale=Egram.scale * 3)
zgram2 = Unit.create((zgram ** 2).dim, name="zgram2", dispname=f"{str(zgram)}^2", scale=zgram.scale * 2)
zgram3 = Unit.create((zgram ** 3).dim, name="zgram3", dispname=f"{str(zgram)}^3", scale=zgram.scale * 3)
Mgram2 = Unit.create((Mgram ** 2).dim, name="Mgram2", dispname=f"{str(Mgram)}^2", scale=Mgram.scale * 2)
Mgram3 = Unit.create((Mgram ** 3).dim, name="Mgram3", dispname=f"{str(Mgram)}^3", scale=Mgram.scale * 3)
kgram2 = Unit.create((kgram ** 2).dim, name="kgram2", dispname=f"{str(kgram)}^2", scale=kgram.scale * 2)
kgram3 = Unit.create((kgram ** 3).dim, name="kgram3", dispname=f"{str(kgram)}^3", scale=kgram.scale * 3)
Ygram2 = Unit.create((Ygram ** 2).dim, name="Ygram2", dispname=f"{str(Ygram)}^2", scale=Ygram.scale * 2)
Ygram3 = Unit.create((Ygram ** 3).dim, name="Ygram3", dispname=f"{str(Ygram)}^3", scale=Ygram.scale * 3)
agramme2 = Unit.create((agramme ** 2).dim, name="agramme2", dispname=f"{str(agramme)}^2", scale=agramme.scale * 2)
agramme3 = Unit.create((agramme ** 3).dim, name="agramme3", dispname=f"{str(agramme)}^3", scale=agramme.scale * 3)
cgramme2 = Unit.create((cgramme ** 2).dim, name="cgramme2", dispname=f"{str(cgramme)}^2", scale=cgramme.scale * 2)
cgramme3 = Unit.create((cgramme ** 3).dim, name="cgramme3", dispname=f"{str(cgramme)}^3", scale=cgramme.scale * 3)
Zgramme2 = Unit.create((Zgramme ** 2).dim, name="Zgramme2", dispname=f"{str(Zgramme)}^2", scale=Zgramme.scale * 2)
Zgramme3 = Unit.create((Zgramme ** 3).dim, name="Zgramme3", dispname=f"{str(Zgramme)}^3", scale=Zgramme.scale * 3)
Pgramme2 = Unit.create((Pgramme ** 2).dim, name="Pgramme2", dispname=f"{str(Pgramme)}^2", scale=Pgramme.scale * 2)
Pgramme3 = Unit.create((Pgramme ** 3).dim, name="Pgramme3", dispname=f"{str(Pgramme)}^3", scale=Pgramme.scale * 3)
dgramme2 = Unit.create((dgramme ** 2).dim, name="dgramme2", dispname=f"{str(dgramme)}^2", scale=dgramme.scale * 2)
dgramme3 = Unit.create((dgramme ** 3).dim, name="dgramme3", dispname=f"{str(dgramme)}^3", scale=dgramme.scale * 3)
Ggramme2 = Unit.create((Ggramme ** 2).dim, name="Ggramme2", dispname=f"{str(Ggramme)}^2", scale=Ggramme.scale * 2)
Ggramme3 = Unit.create((Ggramme ** 3).dim, name="Ggramme3", dispname=f"{str(Ggramme)}^3", scale=Ggramme.scale * 3)
fgramme2 = Unit.create((fgramme ** 2).dim, name="fgramme2", dispname=f"{str(fgramme)}^2", scale=fgramme.scale * 2)
fgramme3 = Unit.create((fgramme ** 3).dim, name="fgramme3", dispname=f"{str(fgramme)}^3", scale=fgramme.scale * 3)
hgramme2 = Unit.create((hgramme ** 2).dim, name="hgramme2", dispname=f"{str(hgramme)}^2", scale=hgramme.scale * 2)
hgramme3 = Unit.create((hgramme ** 3).dim, name="hgramme3", dispname=f"{str(hgramme)}^3", scale=hgramme.scale * 3)
dagramme2 = Unit.create((dagramme ** 2).dim, name="dagramme2", dispname=f"{str(dagramme)}^2",
                        scale=dagramme.scale * 2)
dagramme3 = Unit.create((dagramme ** 3).dim, name="dagramme3", dispname=f"{str(dagramme)}^3",
                        scale=dagramme.scale * 3)
mgramme2 = Unit.create((mgramme ** 2).dim, name="mgramme2", dispname=f"{str(mgramme)}^2", scale=mgramme.scale * 2)
mgramme3 = Unit.create((mgramme ** 3).dim, name="mgramme3", dispname=f"{str(mgramme)}^3", scale=mgramme.scale * 3)
ngramme2 = Unit.create((ngramme ** 2).dim, name="ngramme2", dispname=f"{str(ngramme)}^2", scale=ngramme.scale * 2)
ngramme3 = Unit.create((ngramme ** 3).dim, name="ngramme3", dispname=f"{str(ngramme)}^3", scale=ngramme.scale * 3)
pgramme2 = Unit.create((pgramme ** 2).dim, name="pgramme2", dispname=f"{str(pgramme)}^2", scale=pgramme.scale * 2)
pgramme3 = Unit.create((pgramme ** 3).dim, name="pgramme3", dispname=f"{str(pgramme)}^3", scale=pgramme.scale * 3)
ugramme2 = Unit.create((ugramme ** 2).dim, name="ugramme2", dispname=f"{str(ugramme)}^2", scale=ugramme.scale * 2)
ugramme3 = Unit.create((ugramme ** 3).dim, name="ugramme3", dispname=f"{str(ugramme)}^3", scale=ugramme.scale * 3)
Tgramme2 = Unit.create((Tgramme ** 2).dim, name="Tgramme2", dispname=f"{str(Tgramme)}^2", scale=Tgramme.scale * 2)
Tgramme3 = Unit.create((Tgramme ** 3).dim, name="Tgramme3", dispname=f"{str(Tgramme)}^3", scale=Tgramme.scale * 3)
ygramme2 = Unit.create((ygramme ** 2).dim, name="ygramme2", dispname=f"{str(ygramme)}^2", scale=ygramme.scale * 2)
ygramme3 = Unit.create((ygramme ** 3).dim, name="ygramme3", dispname=f"{str(ygramme)}^3", scale=ygramme.scale * 3)
Egramme2 = Unit.create((Egramme ** 2).dim, name="Egramme2", dispname=f"{str(Egramme)}^2", scale=Egramme.scale * 2)
Egramme3 = Unit.create((Egramme ** 3).dim, name="Egramme3", dispname=f"{str(Egramme)}^3", scale=Egramme.scale * 3)
zgramme2 = Unit.create((zgramme ** 2).dim, name="zgramme2", dispname=f"{str(zgramme)}^2", scale=zgramme.scale * 2)
zgramme3 = Unit.create((zgramme ** 3).dim, name="zgramme3", dispname=f"{str(zgramme)}^3", scale=zgramme.scale * 3)
Mgramme2 = Unit.create((Mgramme ** 2).dim, name="Mgramme2", dispname=f"{str(Mgramme)}^2", scale=Mgramme.scale * 2)
Mgramme3 = Unit.create((Mgramme ** 3).dim, name="Mgramme3", dispname=f"{str(Mgramme)}^3", scale=Mgramme.scale * 3)
kgramme2 = Unit.create((kgramme ** 2).dim, name="kgramme2", dispname=f"{str(kgramme)}^2", scale=kgramme.scale * 2)
kgramme3 = Unit.create((kgramme ** 3).dim, name="kgramme3", dispname=f"{str(kgramme)}^3", scale=kgramme.scale * 3)
Ygramme2 = Unit.create((Ygramme ** 2).dim, name="Ygramme2", dispname=f"{str(Ygramme)}^2", scale=Ygramme.scale * 2)
Ygramme3 = Unit.create((Ygramme ** 3).dim, name="Ygramme3", dispname=f"{str(Ygramme)}^3", scale=Ygramme.scale * 3)
amolar2 = Unit.create((amolar ** 2).dim, name="amolar2", dispname=f"{str(amolar)}^2", scale=amolar.scale * 2)
amolar3 = Unit.create((amolar ** 3).dim, name="amolar3", dispname=f"{str(amolar)}^3", scale=amolar.scale * 3)
cmolar2 = Unit.create((cmolar ** 2).dim, name="cmolar2", dispname=f"{str(cmolar)}^2", scale=cmolar.scale * 2)
cmolar3 = Unit.create((cmolar ** 3).dim, name="cmolar3", dispname=f"{str(cmolar)}^3", scale=cmolar.scale * 3)
Zmolar2 = Unit.create((Zmolar ** 2).dim, name="Zmolar2", dispname=f"{str(Zmolar)}^2", scale=Zmolar.scale * 2)
Zmolar3 = Unit.create((Zmolar ** 3).dim, name="Zmolar3", dispname=f"{str(Zmolar)}^3", scale=Zmolar.scale * 3)
Pmolar2 = Unit.create((Pmolar ** 2).dim, name="Pmolar2", dispname=f"{str(Pmolar)}^2", scale=Pmolar.scale * 2)
Pmolar3 = Unit.create((Pmolar ** 3).dim, name="Pmolar3", dispname=f"{str(Pmolar)}^3", scale=Pmolar.scale * 3)
dmolar2 = Unit.create((dmolar ** 2).dim, name="dmolar2", dispname=f"{str(dmolar)}^2", scale=dmolar.scale * 2)
dmolar3 = Unit.create((dmolar ** 3).dim, name="dmolar3", dispname=f"{str(dmolar)}^3", scale=dmolar.scale * 3)
Gmolar2 = Unit.create((Gmolar ** 2).dim, name="Gmolar2", dispname=f"{str(Gmolar)}^2", scale=Gmolar.scale * 2)
Gmolar3 = Unit.create((Gmolar ** 3).dim, name="Gmolar3", dispname=f"{str(Gmolar)}^3", scale=Gmolar.scale * 3)
fmolar2 = Unit.create((fmolar ** 2).dim, name="fmolar2", dispname=f"{str(fmolar)}^2", scale=fmolar.scale * 2)
fmolar3 = Unit.create((fmolar ** 3).dim, name="fmolar3", dispname=f"{str(fmolar)}^3", scale=fmolar.scale * 3)
hmolar2 = Unit.create((hmolar ** 2).dim, name="hmolar2", dispname=f"{str(hmolar)}^2", scale=hmolar.scale * 2)
hmolar3 = Unit.create((hmolar ** 3).dim, name="hmolar3", dispname=f"{str(hmolar)}^3", scale=hmolar.scale * 3)
damolar2 = Unit.create((damolar ** 2).dim, name="damolar2", dispname=f"{str(damolar)}^2", scale=damolar.scale * 2)
damolar3 = Unit.create((damolar ** 3).dim, name="damolar3", dispname=f"{str(damolar)}^3", scale=damolar.scale * 3)
mmolar2 = Unit.create((mmolar ** 2).dim, name="mmolar2", dispname=f"{str(mmolar)}^2", scale=mmolar.scale * 2)
mmolar3 = Unit.create((mmolar ** 3).dim, name="mmolar3", dispname=f"{str(mmolar)}^3", scale=mmolar.scale * 3)
nmolar2 = Unit.create((nmolar ** 2).dim, name="nmolar2", dispname=f"{str(nmolar)}^2", scale=nmolar.scale * 2)
nmolar3 = Unit.create((nmolar ** 3).dim, name="nmolar3", dispname=f"{str(nmolar)}^3", scale=nmolar.scale * 3)
pmolar2 = Unit.create((pmolar ** 2).dim, name="pmolar2", dispname=f"{str(pmolar)}^2", scale=pmolar.scale * 2)
pmolar3 = Unit.create((pmolar ** 3).dim, name="pmolar3", dispname=f"{str(pmolar)}^3", scale=pmolar.scale * 3)
umolar2 = Unit.create((umolar ** 2).dim, name="umolar2", dispname=f"{str(umolar)}^2", scale=umolar.scale * 2)
umolar3 = Unit.create((umolar ** 3).dim, name="umolar3", dispname=f"{str(umolar)}^3", scale=umolar.scale * 3)
Tmolar2 = Unit.create((Tmolar ** 2).dim, name="Tmolar2", dispname=f"{str(Tmolar)}^2", scale=Tmolar.scale * 2)
Tmolar3 = Unit.create((Tmolar ** 3).dim, name="Tmolar3", dispname=f"{str(Tmolar)}^3", scale=Tmolar.scale * 3)
ymolar2 = Unit.create((ymolar ** 2).dim, name="ymolar2", dispname=f"{str(ymolar)}^2", scale=ymolar.scale * 2)
ymolar3 = Unit.create((ymolar ** 3).dim, name="ymolar3", dispname=f"{str(ymolar)}^3", scale=ymolar.scale * 3)
Emolar2 = Unit.create((Emolar ** 2).dim, name="Emolar2", dispname=f"{str(Emolar)}^2", scale=Emolar.scale * 2)
Emolar3 = Unit.create((Emolar ** 3).dim, name="Emolar3", dispname=f"{str(Emolar)}^3", scale=Emolar.scale * 3)
zmolar2 = Unit.create((zmolar ** 2).dim, name="zmolar2", dispname=f"{str(zmolar)}^2", scale=zmolar.scale * 2)
zmolar3 = Unit.create((zmolar ** 3).dim, name="zmolar3", dispname=f"{str(zmolar)}^3", scale=zmolar.scale * 3)
Mmolar2 = Unit.create((Mmolar ** 2).dim, name="Mmolar2", dispname=f"{str(Mmolar)}^2", scale=Mmolar.scale * 2)
Mmolar3 = Unit.create((Mmolar ** 3).dim, name="Mmolar3", dispname=f"{str(Mmolar)}^3", scale=Mmolar.scale * 3)
kmolar2 = Unit.create((kmolar ** 2).dim, name="kmolar2", dispname=f"{str(kmolar)}^2", scale=kmolar.scale * 2)
kmolar3 = Unit.create((kmolar ** 3).dim, name="kmolar3", dispname=f"{str(kmolar)}^3", scale=kmolar.scale * 3)
Ymolar2 = Unit.create((Ymolar ** 2).dim, name="Ymolar2", dispname=f"{str(Ymolar)}^2", scale=Ymolar.scale * 2)
Ymolar3 = Unit.create((Ymolar ** 3).dim, name="Ymolar3", dispname=f"{str(Ymolar)}^3", scale=Ymolar.scale * 3)
aradian2 = Unit.create((aradian ** 2).dim, name="aradian2", dispname=f"{str(aradian)}^2", scale=aradian.scale * 2)
aradian3 = Unit.create((aradian ** 3).dim, name="aradian3", dispname=f"{str(aradian)}^3", scale=aradian.scale * 3)
cradian2 = Unit.create((cradian ** 2).dim, name="cradian2", dispname=f"{str(cradian)}^2", scale=cradian.scale * 2)
cradian3 = Unit.create((cradian ** 3).dim, name="cradian3", dispname=f"{str(cradian)}^3", scale=cradian.scale * 3)
Zradian2 = Unit.create((Zradian ** 2).dim, name="Zradian2", dispname=f"{str(Zradian)}^2", scale=Zradian.scale * 2)
Zradian3 = Unit.create((Zradian ** 3).dim, name="Zradian3", dispname=f"{str(Zradian)}^3", scale=Zradian.scale * 3)
Pradian2 = Unit.create((Pradian ** 2).dim, name="Pradian2", dispname=f"{str(Pradian)}^2", scale=Pradian.scale * 2)
Pradian3 = Unit.create((Pradian ** 3).dim, name="Pradian3", dispname=f"{str(Pradian)}^3", scale=Pradian.scale * 3)
dradian2 = Unit.create((dradian ** 2).dim, name="dradian2", dispname=f"{str(dradian)}^2", scale=dradian.scale * 2)
dradian3 = Unit.create((dradian ** 3).dim, name="dradian3", dispname=f"{str(dradian)}^3", scale=dradian.scale * 3)
Gradian2 = Unit.create((Gradian ** 2).dim, name="Gradian2", dispname=f"{str(Gradian)}^2", scale=Gradian.scale * 2)
Gradian3 = Unit.create((Gradian ** 3).dim, name="Gradian3", dispname=f"{str(Gradian)}^3", scale=Gradian.scale * 3)
fradian2 = Unit.create((fradian ** 2).dim, name="fradian2", dispname=f"{str(fradian)}^2", scale=fradian.scale * 2)
fradian3 = Unit.create((fradian ** 3).dim, name="fradian3", dispname=f"{str(fradian)}^3", scale=fradian.scale * 3)
hradian2 = Unit.create((hradian ** 2).dim, name="hradian2", dispname=f"{str(hradian)}^2", scale=hradian.scale * 2)
hradian3 = Unit.create((hradian ** 3).dim, name="hradian3", dispname=f"{str(hradian)}^3", scale=hradian.scale * 3)
daradian2 = Unit.create((daradian ** 2).dim, name="daradian2", dispname=f"{str(daradian)}^2",
                        scale=daradian.scale * 2)
daradian3 = Unit.create((daradian ** 3).dim, name="daradian3", dispname=f"{str(daradian)}^3",
                        scale=daradian.scale * 3)
mradian2 = Unit.create((mradian ** 2).dim, name="mradian2", dispname=f"{str(mradian)}^2", scale=mradian.scale * 2)
mradian3 = Unit.create((mradian ** 3).dim, name="mradian3", dispname=f"{str(mradian)}^3", scale=mradian.scale * 3)
nradian2 = Unit.create((nradian ** 2).dim, name="nradian2", dispname=f"{str(nradian)}^2", scale=nradian.scale * 2)
nradian3 = Unit.create((nradian ** 3).dim, name="nradian3", dispname=f"{str(nradian)}^3", scale=nradian.scale * 3)
pradian2 = Unit.create((pradian ** 2).dim, name="pradian2", dispname=f"{str(pradian)}^2", scale=pradian.scale * 2)
pradian3 = Unit.create((pradian ** 3).dim, name="pradian3", dispname=f"{str(pradian)}^3", scale=pradian.scale * 3)
uradian2 = Unit.create((uradian ** 2).dim, name="uradian2", dispname=f"{str(uradian)}^2", scale=uradian.scale * 2)
uradian3 = Unit.create((uradian ** 3).dim, name="uradian3", dispname=f"{str(uradian)}^3", scale=uradian.scale * 3)
Tradian2 = Unit.create((Tradian ** 2).dim, name="Tradian2", dispname=f"{str(Tradian)}^2", scale=Tradian.scale * 2)
Tradian3 = Unit.create((Tradian ** 3).dim, name="Tradian3", dispname=f"{str(Tradian)}^3", scale=Tradian.scale * 3)
yradian2 = Unit.create((yradian ** 2).dim, name="yradian2", dispname=f"{str(yradian)}^2", scale=yradian.scale * 2)
yradian3 = Unit.create((yradian ** 3).dim, name="yradian3", dispname=f"{str(yradian)}^3", scale=yradian.scale * 3)
Eradian2 = Unit.create((Eradian ** 2).dim, name="Eradian2", dispname=f"{str(Eradian)}^2", scale=Eradian.scale * 2)
Eradian3 = Unit.create((Eradian ** 3).dim, name="Eradian3", dispname=f"{str(Eradian)}^3", scale=Eradian.scale * 3)
zradian2 = Unit.create((zradian ** 2).dim, name="zradian2", dispname=f"{str(zradian)}^2", scale=zradian.scale * 2)
zradian3 = Unit.create((zradian ** 3).dim, name="zradian3", dispname=f"{str(zradian)}^3", scale=zradian.scale * 3)
Mradian2 = Unit.create((Mradian ** 2).dim, name="Mradian2", dispname=f"{str(Mradian)}^2", scale=Mradian.scale * 2)
Mradian3 = Unit.create((Mradian ** 3).dim, name="Mradian3", dispname=f"{str(Mradian)}^3", scale=Mradian.scale * 3)
kradian2 = Unit.create((kradian ** 2).dim, name="kradian2", dispname=f"{str(kradian)}^2", scale=kradian.scale * 2)
kradian3 = Unit.create((kradian ** 3).dim, name="kradian3", dispname=f"{str(kradian)}^3", scale=kradian.scale * 3)
Yradian2 = Unit.create((Yradian ** 2).dim, name="Yradian2", dispname=f"{str(Yradian)}^2", scale=Yradian.scale * 2)
Yradian3 = Unit.create((Yradian ** 3).dim, name="Yradian3", dispname=f"{str(Yradian)}^3", scale=Yradian.scale * 3)
asteradian2 = Unit.create((asteradian ** 2).dim, name="asteradian2", dispname=f"{str(asteradian)}^2",
                          scale=asteradian.scale * 2)
asteradian3 = Unit.create((asteradian ** 3).dim, name="asteradian3", dispname=f"{str(asteradian)}^3",
                          scale=asteradian.scale * 3)
csteradian2 = Unit.create((csteradian ** 2).dim, name="csteradian2", dispname=f"{str(csteradian)}^2",
                          scale=csteradian.scale * 2)
csteradian3 = Unit.create((csteradian ** 3).dim, name="csteradian3", dispname=f"{str(csteradian)}^3",
                          scale=csteradian.scale * 3)
Zsteradian2 = Unit.create((Zsteradian ** 2).dim, name="Zsteradian2", dispname=f"{str(Zsteradian)}^2",
                          scale=Zsteradian.scale * 2)
Zsteradian3 = Unit.create((Zsteradian ** 3).dim, name="Zsteradian3", dispname=f"{str(Zsteradian)}^3",
                          scale=Zsteradian.scale * 3)
Psteradian2 = Unit.create((Psteradian ** 2).dim, name="Psteradian2", dispname=f"{str(Psteradian)}^2",
                          scale=Psteradian.scale * 2)
Psteradian3 = Unit.create((Psteradian ** 3).dim, name="Psteradian3", dispname=f"{str(Psteradian)}^3",
                          scale=Psteradian.scale * 3)
dsteradian2 = Unit.create((dsteradian ** 2).dim, name="dsteradian2", dispname=f"{str(dsteradian)}^2",
                          scale=dsteradian.scale * 2)
dsteradian3 = Unit.create((dsteradian ** 3).dim, name="dsteradian3", dispname=f"{str(dsteradian)}^3",
                          scale=dsteradian.scale * 3)
Gsteradian2 = Unit.create((Gsteradian ** 2).dim, name="Gsteradian2", dispname=f"{str(Gsteradian)}^2",
                          scale=Gsteradian.scale * 2)
Gsteradian3 = Unit.create((Gsteradian ** 3).dim, name="Gsteradian3", dispname=f"{str(Gsteradian)}^3",
                          scale=Gsteradian.scale * 3)
fsteradian2 = Unit.create((fsteradian ** 2).dim, name="fsteradian2", dispname=f"{str(fsteradian)}^2",
                          scale=fsteradian.scale * 2)
fsteradian3 = Unit.create((fsteradian ** 3).dim, name="fsteradian3", dispname=f"{str(fsteradian)}^3",
                          scale=fsteradian.scale * 3)
hsteradian2 = Unit.create((hsteradian ** 2).dim, name="hsteradian2", dispname=f"{str(hsteradian)}^2",
                          scale=hsteradian.scale * 2)
hsteradian3 = Unit.create((hsteradian ** 3).dim, name="hsteradian3", dispname=f"{str(hsteradian)}^3",
                          scale=hsteradian.scale * 3)
dasteradian2 = Unit.create((dasteradian ** 2).dim, name="dasteradian2", dispname=f"{str(dasteradian)}^2",
                           scale=dasteradian.scale * 2)
dasteradian3 = Unit.create((dasteradian ** 3).dim, name="dasteradian3", dispname=f"{str(dasteradian)}^3",
                           scale=dasteradian.scale * 3)
msteradian2 = Unit.create((msteradian ** 2).dim, name="msteradian2", dispname=f"{str(msteradian)}^2",
                          scale=msteradian.scale * 2)
msteradian3 = Unit.create((msteradian ** 3).dim, name="msteradian3", dispname=f"{str(msteradian)}^3",
                          scale=msteradian.scale * 3)
nsteradian2 = Unit.create((nsteradian ** 2).dim, name="nsteradian2", dispname=f"{str(nsteradian)}^2",
                          scale=nsteradian.scale * 2)
nsteradian3 = Unit.create((nsteradian ** 3).dim, name="nsteradian3", dispname=f"{str(nsteradian)}^3",
                          scale=nsteradian.scale * 3)
psteradian2 = Unit.create((psteradian ** 2).dim, name="psteradian2", dispname=f"{str(psteradian)}^2",
                          scale=psteradian.scale * 2)
psteradian3 = Unit.create((psteradian ** 3).dim, name="psteradian3", dispname=f"{str(psteradian)}^3",
                          scale=psteradian.scale * 3)
usteradian2 = Unit.create((usteradian ** 2).dim, name="usteradian2", dispname=f"{str(usteradian)}^2",
                          scale=usteradian.scale * 2)
usteradian3 = Unit.create((usteradian ** 3).dim, name="usteradian3", dispname=f"{str(usteradian)}^3",
                          scale=usteradian.scale * 3)
Tsteradian2 = Unit.create((Tsteradian ** 2).dim, name="Tsteradian2", dispname=f"{str(Tsteradian)}^2",
                          scale=Tsteradian.scale * 2)
Tsteradian3 = Unit.create((Tsteradian ** 3).dim, name="Tsteradian3", dispname=f"{str(Tsteradian)}^3",
                          scale=Tsteradian.scale * 3)
ysteradian2 = Unit.create((ysteradian ** 2).dim, name="ysteradian2", dispname=f"{str(ysteradian)}^2",
                          scale=ysteradian.scale * 2)
ysteradian3 = Unit.create((ysteradian ** 3).dim, name="ysteradian3", dispname=f"{str(ysteradian)}^3",
                          scale=ysteradian.scale * 3)
Esteradian2 = Unit.create((Esteradian ** 2).dim, name="Esteradian2", dispname=f"{str(Esteradian)}^2",
                          scale=Esteradian.scale * 2)
Esteradian3 = Unit.create((Esteradian ** 3).dim, name="Esteradian3", dispname=f"{str(Esteradian)}^3",
                          scale=Esteradian.scale * 3)
zsteradian2 = Unit.create((zsteradian ** 2).dim, name="zsteradian2", dispname=f"{str(zsteradian)}^2",
                          scale=zsteradian.scale * 2)
zsteradian3 = Unit.create((zsteradian ** 3).dim, name="zsteradian3", dispname=f"{str(zsteradian)}^3",
                          scale=zsteradian.scale * 3)
Msteradian2 = Unit.create((Msteradian ** 2).dim, name="Msteradian2", dispname=f"{str(Msteradian)}^2",
                          scale=Msteradian.scale * 2)
Msteradian3 = Unit.create((Msteradian ** 3).dim, name="Msteradian3", dispname=f"{str(Msteradian)}^3",
                          scale=Msteradian.scale * 3)
ksteradian2 = Unit.create((ksteradian ** 2).dim, name="ksteradian2", dispname=f"{str(ksteradian)}^2",
                          scale=ksteradian.scale * 2)
ksteradian3 = Unit.create((ksteradian ** 3).dim, name="ksteradian3", dispname=f"{str(ksteradian)}^3",
                          scale=ksteradian.scale * 3)
Ysteradian2 = Unit.create((Ysteradian ** 2).dim, name="Ysteradian2", dispname=f"{str(Ysteradian)}^2",
                          scale=Ysteradian.scale * 2)
Ysteradian3 = Unit.create((Ysteradian ** 3).dim, name="Ysteradian3", dispname=f"{str(Ysteradian)}^3",
                          scale=Ysteradian.scale * 3)
ahertz2 = Unit.create((ahertz ** 2).dim, name="ahertz2", dispname=f"{str(ahertz)}^2", scale=ahertz.scale * 2)
ahertz3 = Unit.create((ahertz ** 3).dim, name="ahertz3", dispname=f"{str(ahertz)}^3", scale=ahertz.scale * 3)
chertz2 = Unit.create((chertz ** 2).dim, name="chertz2", dispname=f"{str(chertz)}^2", scale=chertz.scale * 2)
chertz3 = Unit.create((chertz ** 3).dim, name="chertz3", dispname=f"{str(chertz)}^3", scale=chertz.scale * 3)
Zhertz2 = Unit.create((Zhertz ** 2).dim, name="Zhertz2", dispname=f"{str(Zhertz)}^2", scale=Zhertz.scale * 2)
Zhertz3 = Unit.create((Zhertz ** 3).dim, name="Zhertz3", dispname=f"{str(Zhertz)}^3", scale=Zhertz.scale * 3)
Phertz2 = Unit.create((Phertz ** 2).dim, name="Phertz2", dispname=f"{str(Phertz)}^2", scale=Phertz.scale * 2)
Phertz3 = Unit.create((Phertz ** 3).dim, name="Phertz3", dispname=f"{str(Phertz)}^3", scale=Phertz.scale * 3)
dhertz2 = Unit.create((dhertz ** 2).dim, name="dhertz2", dispname=f"{str(dhertz)}^2", scale=dhertz.scale * 2)
dhertz3 = Unit.create((dhertz ** 3).dim, name="dhertz3", dispname=f"{str(dhertz)}^3", scale=dhertz.scale * 3)
Ghertz2 = Unit.create((Ghertz ** 2).dim, name="Ghertz2", dispname=f"{str(Ghertz)}^2", scale=Ghertz.scale * 2)
Ghertz3 = Unit.create((Ghertz ** 3).dim, name="Ghertz3", dispname=f"{str(Ghertz)}^3", scale=Ghertz.scale * 3)
fhertz2 = Unit.create((fhertz ** 2).dim, name="fhertz2", dispname=f"{str(fhertz)}^2", scale=fhertz.scale * 2)
fhertz3 = Unit.create((fhertz ** 3).dim, name="fhertz3", dispname=f"{str(fhertz)}^3", scale=fhertz.scale * 3)
hhertz2 = Unit.create((hhertz ** 2).dim, name="hhertz2", dispname=f"{str(hhertz)}^2", scale=hhertz.scale * 2)
hhertz3 = Unit.create((hhertz ** 3).dim, name="hhertz3", dispname=f"{str(hhertz)}^3", scale=hhertz.scale * 3)
dahertz2 = Unit.create((dahertz ** 2).dim, name="dahertz2", dispname=f"{str(dahertz)}^2", scale=dahertz.scale * 2)
dahertz3 = Unit.create((dahertz ** 3).dim, name="dahertz3", dispname=f"{str(dahertz)}^3", scale=dahertz.scale * 3)
mhertz2 = Unit.create((mhertz ** 2).dim, name="mhertz2", dispname=f"{str(mhertz)}^2", scale=mhertz.scale * 2)
mhertz3 = Unit.create((mhertz ** 3).dim, name="mhertz3", dispname=f"{str(mhertz)}^3", scale=mhertz.scale * 3)
nhertz2 = Unit.create((nhertz ** 2).dim, name="nhertz2", dispname=f"{str(nhertz)}^2", scale=nhertz.scale * 2)
nhertz3 = Unit.create((nhertz ** 3).dim, name="nhertz3", dispname=f"{str(nhertz)}^3", scale=nhertz.scale * 3)
phertz2 = Unit.create((phertz ** 2).dim, name="phertz2", dispname=f"{str(phertz)}^2", scale=phertz.scale * 2)
phertz3 = Unit.create((phertz ** 3).dim, name="phertz3", dispname=f"{str(phertz)}^3", scale=phertz.scale * 3)
uhertz2 = Unit.create((uhertz ** 2).dim, name="uhertz2", dispname=f"{str(uhertz)}^2", scale=uhertz.scale * 2)
uhertz3 = Unit.create((uhertz ** 3).dim, name="uhertz3", dispname=f"{str(uhertz)}^3", scale=uhertz.scale * 3)
Thertz2 = Unit.create((Thertz ** 2).dim, name="Thertz2", dispname=f"{str(Thertz)}^2", scale=Thertz.scale * 2)
Thertz3 = Unit.create((Thertz ** 3).dim, name="Thertz3", dispname=f"{str(Thertz)}^3", scale=Thertz.scale * 3)
yhertz2 = Unit.create((yhertz ** 2).dim, name="yhertz2", dispname=f"{str(yhertz)}^2", scale=yhertz.scale * 2)
yhertz3 = Unit.create((yhertz ** 3).dim, name="yhertz3", dispname=f"{str(yhertz)}^3", scale=yhertz.scale * 3)
Ehertz2 = Unit.create((Ehertz ** 2).dim, name="Ehertz2", dispname=f"{str(Ehertz)}^2", scale=Ehertz.scale * 2)
Ehertz3 = Unit.create((Ehertz ** 3).dim, name="Ehertz3", dispname=f"{str(Ehertz)}^3", scale=Ehertz.scale * 3)
zhertz2 = Unit.create((zhertz ** 2).dim, name="zhertz2", dispname=f"{str(zhertz)}^2", scale=zhertz.scale * 2)
zhertz3 = Unit.create((zhertz ** 3).dim, name="zhertz3", dispname=f"{str(zhertz)}^3", scale=zhertz.scale * 3)
Mhertz2 = Unit.create((Mhertz ** 2).dim, name="Mhertz2", dispname=f"{str(Mhertz)}^2", scale=Mhertz.scale * 2)
Mhertz3 = Unit.create((Mhertz ** 3).dim, name="Mhertz3", dispname=f"{str(Mhertz)}^3", scale=Mhertz.scale * 3)
khertz2 = Unit.create((khertz ** 2).dim, name="khertz2", dispname=f"{str(khertz)}^2", scale=khertz.scale * 2)
khertz3 = Unit.create((khertz ** 3).dim, name="khertz3", dispname=f"{str(khertz)}^3", scale=khertz.scale * 3)
Yhertz2 = Unit.create((Yhertz ** 2).dim, name="Yhertz2", dispname=f"{str(Yhertz)}^2", scale=Yhertz.scale * 2)
Yhertz3 = Unit.create((Yhertz ** 3).dim, name="Yhertz3", dispname=f"{str(Yhertz)}^3", scale=Yhertz.scale * 3)
anewton2 = Unit.create((anewton ** 2).dim, name="anewton2", dispname=f"{str(anewton)}^2", scale=anewton.scale * 2)
anewton3 = Unit.create((anewton ** 3).dim, name="anewton3", dispname=f"{str(anewton)}^3", scale=anewton.scale * 3)
cnewton2 = Unit.create((cnewton ** 2).dim, name="cnewton2", dispname=f"{str(cnewton)}^2", scale=cnewton.scale * 2)
cnewton3 = Unit.create((cnewton ** 3).dim, name="cnewton3", dispname=f"{str(cnewton)}^3", scale=cnewton.scale * 3)
Znewton2 = Unit.create((Znewton ** 2).dim, name="Znewton2", dispname=f"{str(Znewton)}^2", scale=Znewton.scale * 2)
Znewton3 = Unit.create((Znewton ** 3).dim, name="Znewton3", dispname=f"{str(Znewton)}^3", scale=Znewton.scale * 3)
Pnewton2 = Unit.create((Pnewton ** 2).dim, name="Pnewton2", dispname=f"{str(Pnewton)}^2", scale=Pnewton.scale * 2)
Pnewton3 = Unit.create((Pnewton ** 3).dim, name="Pnewton3", dispname=f"{str(Pnewton)}^3", scale=Pnewton.scale * 3)
dnewton2 = Unit.create((dnewton ** 2).dim, name="dnewton2", dispname=f"{str(dnewton)}^2", scale=dnewton.scale * 2)
dnewton3 = Unit.create((dnewton ** 3).dim, name="dnewton3", dispname=f"{str(dnewton)}^3", scale=dnewton.scale * 3)
Gnewton2 = Unit.create((Gnewton ** 2).dim, name="Gnewton2", dispname=f"{str(Gnewton)}^2", scale=Gnewton.scale * 2)
Gnewton3 = Unit.create((Gnewton ** 3).dim, name="Gnewton3", dispname=f"{str(Gnewton)}^3", scale=Gnewton.scale * 3)
fnewton2 = Unit.create((fnewton ** 2).dim, name="fnewton2", dispname=f"{str(fnewton)}^2", scale=fnewton.scale * 2)
fnewton3 = Unit.create((fnewton ** 3).dim, name="fnewton3", dispname=f"{str(fnewton)}^3", scale=fnewton.scale * 3)
hnewton2 = Unit.create((hnewton ** 2).dim, name="hnewton2", dispname=f"{str(hnewton)}^2", scale=hnewton.scale * 2)
hnewton3 = Unit.create((hnewton ** 3).dim, name="hnewton3", dispname=f"{str(hnewton)}^3", scale=hnewton.scale * 3)
danewton2 = Unit.create((danewton ** 2).dim, name="danewton2", dispname=f"{str(danewton)}^2",
                        scale=danewton.scale * 2)
danewton3 = Unit.create((danewton ** 3).dim, name="danewton3", dispname=f"{str(danewton)}^3",
                        scale=danewton.scale * 3)
mnewton2 = Unit.create((mnewton ** 2).dim, name="mnewton2", dispname=f"{str(mnewton)}^2", scale=mnewton.scale * 2)
mnewton3 = Unit.create((mnewton ** 3).dim, name="mnewton3", dispname=f"{str(mnewton)}^3", scale=mnewton.scale * 3)
nnewton2 = Unit.create((nnewton ** 2).dim, name="nnewton2", dispname=f"{str(nnewton)}^2", scale=nnewton.scale * 2)
nnewton3 = Unit.create((nnewton ** 3).dim, name="nnewton3", dispname=f"{str(nnewton)}^3", scale=nnewton.scale * 3)
pnewton2 = Unit.create((pnewton ** 2).dim, name="pnewton2", dispname=f"{str(pnewton)}^2", scale=pnewton.scale * 2)
pnewton3 = Unit.create((pnewton ** 3).dim, name="pnewton3", dispname=f"{str(pnewton)}^3", scale=pnewton.scale * 3)
unewton2 = Unit.create((unewton ** 2).dim, name="unewton2", dispname=f"{str(unewton)}^2", scale=unewton.scale * 2)
unewton3 = Unit.create((unewton ** 3).dim, name="unewton3", dispname=f"{str(unewton)}^3", scale=unewton.scale * 3)
Tnewton2 = Unit.create((Tnewton ** 2).dim, name="Tnewton2", dispname=f"{str(Tnewton)}^2", scale=Tnewton.scale * 2)
Tnewton3 = Unit.create((Tnewton ** 3).dim, name="Tnewton3", dispname=f"{str(Tnewton)}^3", scale=Tnewton.scale * 3)
ynewton2 = Unit.create((ynewton ** 2).dim, name="ynewton2", dispname=f"{str(ynewton)}^2", scale=ynewton.scale * 2)
ynewton3 = Unit.create((ynewton ** 3).dim, name="ynewton3", dispname=f"{str(ynewton)}^3", scale=ynewton.scale * 3)
Enewton2 = Unit.create((Enewton ** 2).dim, name="Enewton2", dispname=f"{str(Enewton)}^2", scale=Enewton.scale * 2)
Enewton3 = Unit.create((Enewton ** 3).dim, name="Enewton3", dispname=f"{str(Enewton)}^3", scale=Enewton.scale * 3)
znewton2 = Unit.create((znewton ** 2).dim, name="znewton2", dispname=f"{str(znewton)}^2", scale=znewton.scale * 2)
znewton3 = Unit.create((znewton ** 3).dim, name="znewton3", dispname=f"{str(znewton)}^3", scale=znewton.scale * 3)
Mnewton2 = Unit.create((Mnewton ** 2).dim, name="Mnewton2", dispname=f"{str(Mnewton)}^2", scale=Mnewton.scale * 2)
Mnewton3 = Unit.create((Mnewton ** 3).dim, name="Mnewton3", dispname=f"{str(Mnewton)}^3", scale=Mnewton.scale * 3)
knewton2 = Unit.create((knewton ** 2).dim, name="knewton2", dispname=f"{str(knewton)}^2", scale=knewton.scale * 2)
knewton3 = Unit.create((knewton ** 3).dim, name="knewton3", dispname=f"{str(knewton)}^3", scale=knewton.scale * 3)
Ynewton2 = Unit.create((Ynewton ** 2).dim, name="Ynewton2", dispname=f"{str(Ynewton)}^2", scale=Ynewton.scale * 2)
Ynewton3 = Unit.create((Ynewton ** 3).dim, name="Ynewton3", dispname=f"{str(Ynewton)}^3", scale=Ynewton.scale * 3)
apascal2 = Unit.create((apascal ** 2).dim, name="apascal2", dispname=f"{str(apascal)}^2", scale=apascal.scale * 2)
apascal3 = Unit.create((apascal ** 3).dim, name="apascal3", dispname=f"{str(apascal)}^3", scale=apascal.scale * 3)
cpascal2 = Unit.create((cpascal ** 2).dim, name="cpascal2", dispname=f"{str(cpascal)}^2", scale=cpascal.scale * 2)
cpascal3 = Unit.create((cpascal ** 3).dim, name="cpascal3", dispname=f"{str(cpascal)}^3", scale=cpascal.scale * 3)
Zpascal2 = Unit.create((Zpascal ** 2).dim, name="Zpascal2", dispname=f"{str(Zpascal)}^2", scale=Zpascal.scale * 2)
Zpascal3 = Unit.create((Zpascal ** 3).dim, name="Zpascal3", dispname=f"{str(Zpascal)}^3", scale=Zpascal.scale * 3)
Ppascal2 = Unit.create((Ppascal ** 2).dim, name="Ppascal2", dispname=f"{str(Ppascal)}^2", scale=Ppascal.scale * 2)
Ppascal3 = Unit.create((Ppascal ** 3).dim, name="Ppascal3", dispname=f"{str(Ppascal)}^3", scale=Ppascal.scale * 3)
dpascal2 = Unit.create((dpascal ** 2).dim, name="dpascal2", dispname=f"{str(dpascal)}^2", scale=dpascal.scale * 2)
dpascal3 = Unit.create((dpascal ** 3).dim, name="dpascal3", dispname=f"{str(dpascal)}^3", scale=dpascal.scale * 3)
Gpascal2 = Unit.create((Gpascal ** 2).dim, name="Gpascal2", dispname=f"{str(Gpascal)}^2", scale=Gpascal.scale * 2)
Gpascal3 = Unit.create((Gpascal ** 3).dim, name="Gpascal3", dispname=f"{str(Gpascal)}^3", scale=Gpascal.scale * 3)
fpascal2 = Unit.create((fpascal ** 2).dim, name="fpascal2", dispname=f"{str(fpascal)}^2", scale=fpascal.scale * 2)
fpascal3 = Unit.create((fpascal ** 3).dim, name="fpascal3", dispname=f"{str(fpascal)}^3", scale=fpascal.scale * 3)
hpascal2 = Unit.create((hpascal ** 2).dim, name="hpascal2", dispname=f"{str(hpascal)}^2", scale=hpascal.scale * 2)
hpascal3 = Unit.create((hpascal ** 3).dim, name="hpascal3", dispname=f"{str(hpascal)}^3", scale=hpascal.scale * 3)
dapascal2 = Unit.create((dapascal ** 2).dim, name="dapascal2", dispname=f"{str(dapascal)}^2",
                        scale=dapascal.scale * 2)
dapascal3 = Unit.create((dapascal ** 3).dim, name="dapascal3", dispname=f"{str(dapascal)}^3",
                        scale=dapascal.scale * 3)
mpascal2 = Unit.create((mpascal ** 2).dim, name="mpascal2", dispname=f"{str(mpascal)}^2", scale=mpascal.scale * 2)
mpascal3 = Unit.create((mpascal ** 3).dim, name="mpascal3", dispname=f"{str(mpascal)}^3", scale=mpascal.scale * 3)
npascal2 = Unit.create((npascal ** 2).dim, name="npascal2", dispname=f"{str(npascal)}^2", scale=npascal.scale * 2)
npascal3 = Unit.create((npascal ** 3).dim, name="npascal3", dispname=f"{str(npascal)}^3", scale=npascal.scale * 3)
ppascal2 = Unit.create((ppascal ** 2).dim, name="ppascal2", dispname=f"{str(ppascal)}^2", scale=ppascal.scale * 2)
ppascal3 = Unit.create((ppascal ** 3).dim, name="ppascal3", dispname=f"{str(ppascal)}^3", scale=ppascal.scale * 3)
upascal2 = Unit.create((upascal ** 2).dim, name="upascal2", dispname=f"{str(upascal)}^2", scale=upascal.scale * 2)
upascal3 = Unit.create((upascal ** 3).dim, name="upascal3", dispname=f"{str(upascal)}^3", scale=upascal.scale * 3)
Tpascal2 = Unit.create((Tpascal ** 2).dim, name="Tpascal2", dispname=f"{str(Tpascal)}^2", scale=Tpascal.scale * 2)
Tpascal3 = Unit.create((Tpascal ** 3).dim, name="Tpascal3", dispname=f"{str(Tpascal)}^3", scale=Tpascal.scale * 3)
ypascal2 = Unit.create((ypascal ** 2).dim, name="ypascal2", dispname=f"{str(ypascal)}^2", scale=ypascal.scale * 2)
ypascal3 = Unit.create((ypascal ** 3).dim, name="ypascal3", dispname=f"{str(ypascal)}^3", scale=ypascal.scale * 3)
Epascal2 = Unit.create((Epascal ** 2).dim, name="Epascal2", dispname=f"{str(Epascal)}^2", scale=Epascal.scale * 2)
Epascal3 = Unit.create((Epascal ** 3).dim, name="Epascal3", dispname=f"{str(Epascal)}^3", scale=Epascal.scale * 3)
zpascal2 = Unit.create((zpascal ** 2).dim, name="zpascal2", dispname=f"{str(zpascal)}^2", scale=zpascal.scale * 2)
zpascal3 = Unit.create((zpascal ** 3).dim, name="zpascal3", dispname=f"{str(zpascal)}^3", scale=zpascal.scale * 3)
Mpascal2 = Unit.create((Mpascal ** 2).dim, name="Mpascal2", dispname=f"{str(Mpascal)}^2", scale=Mpascal.scale * 2)
Mpascal3 = Unit.create((Mpascal ** 3).dim, name="Mpascal3", dispname=f"{str(Mpascal)}^3", scale=Mpascal.scale * 3)
kpascal2 = Unit.create((kpascal ** 2).dim, name="kpascal2", dispname=f"{str(kpascal)}^2", scale=kpascal.scale * 2)
kpascal3 = Unit.create((kpascal ** 3).dim, name="kpascal3", dispname=f"{str(kpascal)}^3", scale=kpascal.scale * 3)
Ypascal2 = Unit.create((Ypascal ** 2).dim, name="Ypascal2", dispname=f"{str(Ypascal)}^2", scale=Ypascal.scale * 2)
Ypascal3 = Unit.create((Ypascal ** 3).dim, name="Ypascal3", dispname=f"{str(Ypascal)}^3", scale=Ypascal.scale * 3)
ajoule2 = Unit.create((ajoule ** 2).dim, name="ajoule2", dispname=f"{str(ajoule)}^2", scale=ajoule.scale * 2)
ajoule3 = Unit.create((ajoule ** 3).dim, name="ajoule3", dispname=f"{str(ajoule)}^3", scale=ajoule.scale * 3)
cjoule2 = Unit.create((cjoule ** 2).dim, name="cjoule2", dispname=f"{str(cjoule)}^2", scale=cjoule.scale * 2)
cjoule3 = Unit.create((cjoule ** 3).dim, name="cjoule3", dispname=f"{str(cjoule)}^3", scale=cjoule.scale * 3)
Zjoule2 = Unit.create((Zjoule ** 2).dim, name="Zjoule2", dispname=f"{str(Zjoule)}^2", scale=Zjoule.scale * 2)
Zjoule3 = Unit.create((Zjoule ** 3).dim, name="Zjoule3", dispname=f"{str(Zjoule)}^3", scale=Zjoule.scale * 3)
Pjoule2 = Unit.create((Pjoule ** 2).dim, name="Pjoule2", dispname=f"{str(Pjoule)}^2", scale=Pjoule.scale * 2)
Pjoule3 = Unit.create((Pjoule ** 3).dim, name="Pjoule3", dispname=f"{str(Pjoule)}^3", scale=Pjoule.scale * 3)
djoule2 = Unit.create((djoule ** 2).dim, name="djoule2", dispname=f"{str(djoule)}^2", scale=djoule.scale * 2)
djoule3 = Unit.create((djoule ** 3).dim, name="djoule3", dispname=f"{str(djoule)}^3", scale=djoule.scale * 3)
Gjoule2 = Unit.create((Gjoule ** 2).dim, name="Gjoule2", dispname=f"{str(Gjoule)}^2", scale=Gjoule.scale * 2)
Gjoule3 = Unit.create((Gjoule ** 3).dim, name="Gjoule3", dispname=f"{str(Gjoule)}^3", scale=Gjoule.scale * 3)
fjoule2 = Unit.create((fjoule ** 2).dim, name="fjoule2", dispname=f"{str(fjoule)}^2", scale=fjoule.scale * 2)
fjoule3 = Unit.create((fjoule ** 3).dim, name="fjoule3", dispname=f"{str(fjoule)}^3", scale=fjoule.scale * 3)
hjoule2 = Unit.create((hjoule ** 2).dim, name="hjoule2", dispname=f"{str(hjoule)}^2", scale=hjoule.scale * 2)
hjoule3 = Unit.create((hjoule ** 3).dim, name="hjoule3", dispname=f"{str(hjoule)}^3", scale=hjoule.scale * 3)
dajoule2 = Unit.create((dajoule ** 2).dim, name="dajoule2", dispname=f"{str(dajoule)}^2", scale=dajoule.scale * 2)
dajoule3 = Unit.create((dajoule ** 3).dim, name="dajoule3", dispname=f"{str(dajoule)}^3", scale=dajoule.scale * 3)
mjoule2 = Unit.create((mjoule ** 2).dim, name="mjoule2", dispname=f"{str(mjoule)}^2", scale=mjoule.scale * 2)
mjoule3 = Unit.create((mjoule ** 3).dim, name="mjoule3", dispname=f"{str(mjoule)}^3", scale=mjoule.scale * 3)
njoule2 = Unit.create((njoule ** 2).dim, name="njoule2", dispname=f"{str(njoule)}^2", scale=njoule.scale * 2)
njoule3 = Unit.create((njoule ** 3).dim, name="njoule3", dispname=f"{str(njoule)}^3", scale=njoule.scale * 3)
pjoule2 = Unit.create((pjoule ** 2).dim, name="pjoule2", dispname=f"{str(pjoule)}^2", scale=pjoule.scale * 2)
pjoule3 = Unit.create((pjoule ** 3).dim, name="pjoule3", dispname=f"{str(pjoule)}^3", scale=pjoule.scale * 3)
ujoule2 = Unit.create((ujoule ** 2).dim, name="ujoule2", dispname=f"{str(ujoule)}^2", scale=ujoule.scale * 2)
ujoule3 = Unit.create((ujoule ** 3).dim, name="ujoule3", dispname=f"{str(ujoule)}^3", scale=ujoule.scale * 3)
Tjoule2 = Unit.create((Tjoule ** 2).dim, name="Tjoule2", dispname=f"{str(Tjoule)}^2", scale=Tjoule.scale * 2)
Tjoule3 = Unit.create((Tjoule ** 3).dim, name="Tjoule3", dispname=f"{str(Tjoule)}^3", scale=Tjoule.scale * 3)
yjoule2 = Unit.create((yjoule ** 2).dim, name="yjoule2", dispname=f"{str(yjoule)}^2", scale=yjoule.scale * 2)
yjoule3 = Unit.create((yjoule ** 3).dim, name="yjoule3", dispname=f"{str(yjoule)}^3", scale=yjoule.scale * 3)
Ejoule2 = Unit.create((Ejoule ** 2).dim, name="Ejoule2", dispname=f"{str(Ejoule)}^2", scale=Ejoule.scale * 2)
Ejoule3 = Unit.create((Ejoule ** 3).dim, name="Ejoule3", dispname=f"{str(Ejoule)}^3", scale=Ejoule.scale * 3)
zjoule2 = Unit.create((zjoule ** 2).dim, name="zjoule2", dispname=f"{str(zjoule)}^2", scale=zjoule.scale * 2)
zjoule3 = Unit.create((zjoule ** 3).dim, name="zjoule3", dispname=f"{str(zjoule)}^3", scale=zjoule.scale * 3)
Mjoule2 = Unit.create((Mjoule ** 2).dim, name="Mjoule2", dispname=f"{str(Mjoule)}^2", scale=Mjoule.scale * 2)
Mjoule3 = Unit.create((Mjoule ** 3).dim, name="Mjoule3", dispname=f"{str(Mjoule)}^3", scale=Mjoule.scale * 3)
kjoule2 = Unit.create((kjoule ** 2).dim, name="kjoule2", dispname=f"{str(kjoule)}^2", scale=kjoule.scale * 2)
kjoule3 = Unit.create((kjoule ** 3).dim, name="kjoule3", dispname=f"{str(kjoule)}^3", scale=kjoule.scale * 3)
Yjoule2 = Unit.create((Yjoule ** 2).dim, name="Yjoule2", dispname=f"{str(Yjoule)}^2", scale=Yjoule.scale * 2)
Yjoule3 = Unit.create((Yjoule ** 3).dim, name="Yjoule3", dispname=f"{str(Yjoule)}^3", scale=Yjoule.scale * 3)
awatt2 = Unit.create((awatt ** 2).dim, name="awatt2", dispname=f"{str(awatt)}^2", scale=awatt.scale * 2)
awatt3 = Unit.create((awatt ** 3).dim, name="awatt3", dispname=f"{str(awatt)}^3", scale=awatt.scale * 3)
cwatt2 = Unit.create((cwatt ** 2).dim, name="cwatt2", dispname=f"{str(cwatt)}^2", scale=cwatt.scale * 2)
cwatt3 = Unit.create((cwatt ** 3).dim, name="cwatt3", dispname=f"{str(cwatt)}^3", scale=cwatt.scale * 3)
Zwatt2 = Unit.create((Zwatt ** 2).dim, name="Zwatt2", dispname=f"{str(Zwatt)}^2", scale=Zwatt.scale * 2)
Zwatt3 = Unit.create((Zwatt ** 3).dim, name="Zwatt3", dispname=f"{str(Zwatt)}^3", scale=Zwatt.scale * 3)
Pwatt2 = Unit.create((Pwatt ** 2).dim, name="Pwatt2", dispname=f"{str(Pwatt)}^2", scale=Pwatt.scale * 2)
Pwatt3 = Unit.create((Pwatt ** 3).dim, name="Pwatt3", dispname=f"{str(Pwatt)}^3", scale=Pwatt.scale * 3)
dwatt2 = Unit.create((dwatt ** 2).dim, name="dwatt2", dispname=f"{str(dwatt)}^2", scale=dwatt.scale * 2)
dwatt3 = Unit.create((dwatt ** 3).dim, name="dwatt3", dispname=f"{str(dwatt)}^3", scale=dwatt.scale * 3)
Gwatt2 = Unit.create((Gwatt ** 2).dim, name="Gwatt2", dispname=f"{str(Gwatt)}^2", scale=Gwatt.scale * 2)
Gwatt3 = Unit.create((Gwatt ** 3).dim, name="Gwatt3", dispname=f"{str(Gwatt)}^3", scale=Gwatt.scale * 3)
fwatt2 = Unit.create((fwatt ** 2).dim, name="fwatt2", dispname=f"{str(fwatt)}^2", scale=fwatt.scale * 2)
fwatt3 = Unit.create((fwatt ** 3).dim, name="fwatt3", dispname=f"{str(fwatt)}^3", scale=fwatt.scale * 3)
hwatt2 = Unit.create((hwatt ** 2).dim, name="hwatt2", dispname=f"{str(hwatt)}^2", scale=hwatt.scale * 2)
hwatt3 = Unit.create((hwatt ** 3).dim, name="hwatt3", dispname=f"{str(hwatt)}^3", scale=hwatt.scale * 3)
dawatt2 = Unit.create((dawatt ** 2).dim, name="dawatt2", dispname=f"{str(dawatt)}^2", scale=dawatt.scale * 2)
dawatt3 = Unit.create((dawatt ** 3).dim, name="dawatt3", dispname=f"{str(dawatt)}^3", scale=dawatt.scale * 3)
mwatt2 = Unit.create((mwatt ** 2).dim, name="mwatt2", dispname=f"{str(mwatt)}^2", scale=mwatt.scale * 2)
mwatt3 = Unit.create((mwatt ** 3).dim, name="mwatt3", dispname=f"{str(mwatt)}^3", scale=mwatt.scale * 3)
nwatt2 = Unit.create((nwatt ** 2).dim, name="nwatt2", dispname=f"{str(nwatt)}^2", scale=nwatt.scale * 2)
nwatt3 = Unit.create((nwatt ** 3).dim, name="nwatt3", dispname=f"{str(nwatt)}^3", scale=nwatt.scale * 3)
pwatt2 = Unit.create((pwatt ** 2).dim, name="pwatt2", dispname=f"{str(pwatt)}^2", scale=pwatt.scale * 2)
pwatt3 = Unit.create((pwatt ** 3).dim, name="pwatt3", dispname=f"{str(pwatt)}^3", scale=pwatt.scale * 3)
uwatt2 = Unit.create((uwatt ** 2).dim, name="uwatt2", dispname=f"{str(uwatt)}^2", scale=uwatt.scale * 2)
uwatt3 = Unit.create((uwatt ** 3).dim, name="uwatt3", dispname=f"{str(uwatt)}^3", scale=uwatt.scale * 3)
Twatt2 = Unit.create((Twatt ** 2).dim, name="Twatt2", dispname=f"{str(Twatt)}^2", scale=Twatt.scale * 2)
Twatt3 = Unit.create((Twatt ** 3).dim, name="Twatt3", dispname=f"{str(Twatt)}^3", scale=Twatt.scale * 3)
ywatt2 = Unit.create((ywatt ** 2).dim, name="ywatt2", dispname=f"{str(ywatt)}^2", scale=ywatt.scale * 2)
ywatt3 = Unit.create((ywatt ** 3).dim, name="ywatt3", dispname=f"{str(ywatt)}^3", scale=ywatt.scale * 3)
Ewatt2 = Unit.create((Ewatt ** 2).dim, name="Ewatt2", dispname=f"{str(Ewatt)}^2", scale=Ewatt.scale * 2)
Ewatt3 = Unit.create((Ewatt ** 3).dim, name="Ewatt3", dispname=f"{str(Ewatt)}^3", scale=Ewatt.scale * 3)
zwatt2 = Unit.create((zwatt ** 2).dim, name="zwatt2", dispname=f"{str(zwatt)}^2", scale=zwatt.scale * 2)
zwatt3 = Unit.create((zwatt ** 3).dim, name="zwatt3", dispname=f"{str(zwatt)}^3", scale=zwatt.scale * 3)
Mwatt2 = Unit.create((Mwatt ** 2).dim, name="Mwatt2", dispname=f"{str(Mwatt)}^2", scale=Mwatt.scale * 2)
Mwatt3 = Unit.create((Mwatt ** 3).dim, name="Mwatt3", dispname=f"{str(Mwatt)}^3", scale=Mwatt.scale * 3)
kwatt2 = Unit.create((kwatt ** 2).dim, name="kwatt2", dispname=f"{str(kwatt)}^2", scale=kwatt.scale * 2)
kwatt3 = Unit.create((kwatt ** 3).dim, name="kwatt3", dispname=f"{str(kwatt)}^3", scale=kwatt.scale * 3)
Ywatt2 = Unit.create((Ywatt ** 2).dim, name="Ywatt2", dispname=f"{str(Ywatt)}^2", scale=Ywatt.scale * 2)
Ywatt3 = Unit.create((Ywatt ** 3).dim, name="Ywatt3", dispname=f"{str(Ywatt)}^3", scale=Ywatt.scale * 3)
acoulomb2 = Unit.create((acoulomb ** 2).dim, name="acoulomb2", dispname=f"{str(acoulomb)}^2",
                        scale=acoulomb.scale * 2)
acoulomb3 = Unit.create((acoulomb ** 3).dim, name="acoulomb3", dispname=f"{str(acoulomb)}^3",
                        scale=acoulomb.scale * 3)
ccoulomb2 = Unit.create((ccoulomb ** 2).dim, name="ccoulomb2", dispname=f"{str(ccoulomb)}^2",
                        scale=ccoulomb.scale * 2)
ccoulomb3 = Unit.create((ccoulomb ** 3).dim, name="ccoulomb3", dispname=f"{str(ccoulomb)}^3",
                        scale=ccoulomb.scale * 3)
Zcoulomb2 = Unit.create((Zcoulomb ** 2).dim, name="Zcoulomb2", dispname=f"{str(Zcoulomb)}^2",
                        scale=Zcoulomb.scale * 2)
Zcoulomb3 = Unit.create((Zcoulomb ** 3).dim, name="Zcoulomb3", dispname=f"{str(Zcoulomb)}^3",
                        scale=Zcoulomb.scale * 3)
Pcoulomb2 = Unit.create((Pcoulomb ** 2).dim, name="Pcoulomb2", dispname=f"{str(Pcoulomb)}^2",
                        scale=Pcoulomb.scale * 2)
Pcoulomb3 = Unit.create((Pcoulomb ** 3).dim, name="Pcoulomb3", dispname=f"{str(Pcoulomb)}^3",
                        scale=Pcoulomb.scale * 3)
dcoulomb2 = Unit.create((dcoulomb ** 2).dim, name="dcoulomb2", dispname=f"{str(dcoulomb)}^2",
                        scale=dcoulomb.scale * 2)
dcoulomb3 = Unit.create((dcoulomb ** 3).dim, name="dcoulomb3", dispname=f"{str(dcoulomb)}^3",
                        scale=dcoulomb.scale * 3)
Gcoulomb2 = Unit.create((Gcoulomb ** 2).dim, name="Gcoulomb2", dispname=f"{str(Gcoulomb)}^2",
                        scale=Gcoulomb.scale * 2)
Gcoulomb3 = Unit.create((Gcoulomb ** 3).dim, name="Gcoulomb3", dispname=f"{str(Gcoulomb)}^3",
                        scale=Gcoulomb.scale * 3)
fcoulomb2 = Unit.create((fcoulomb ** 2).dim, name="fcoulomb2", dispname=f"{str(fcoulomb)}^2",
                        scale=fcoulomb.scale * 2)
fcoulomb3 = Unit.create((fcoulomb ** 3).dim, name="fcoulomb3", dispname=f"{str(fcoulomb)}^3",
                        scale=fcoulomb.scale * 3)
hcoulomb2 = Unit.create((hcoulomb ** 2).dim, name="hcoulomb2", dispname=f"{str(hcoulomb)}^2",
                        scale=hcoulomb.scale * 2)
hcoulomb3 = Unit.create((hcoulomb ** 3).dim, name="hcoulomb3", dispname=f"{str(hcoulomb)}^3",
                        scale=hcoulomb.scale * 3)
dacoulomb2 = Unit.create((dacoulomb ** 2).dim, name="dacoulomb2", dispname=f"{str(dacoulomb)}^2",
                         scale=dacoulomb.scale * 2)
dacoulomb3 = Unit.create((dacoulomb ** 3).dim, name="dacoulomb3", dispname=f"{str(dacoulomb)}^3",
                         scale=dacoulomb.scale * 3)
mcoulomb2 = Unit.create((mcoulomb ** 2).dim, name="mcoulomb2", dispname=f"{str(mcoulomb)}^2",
                        scale=mcoulomb.scale * 2)
mcoulomb3 = Unit.create((mcoulomb ** 3).dim, name="mcoulomb3", dispname=f"{str(mcoulomb)}^3",
                        scale=mcoulomb.scale * 3)
ncoulomb2 = Unit.create((ncoulomb ** 2).dim, name="ncoulomb2", dispname=f"{str(ncoulomb)}^2",
                        scale=ncoulomb.scale * 2)
ncoulomb3 = Unit.create((ncoulomb ** 3).dim, name="ncoulomb3", dispname=f"{str(ncoulomb)}^3",
                        scale=ncoulomb.scale * 3)
pcoulomb2 = Unit.create((pcoulomb ** 2).dim, name="pcoulomb2", dispname=f"{str(pcoulomb)}^2",
                        scale=pcoulomb.scale * 2)
pcoulomb3 = Unit.create((pcoulomb ** 3).dim, name="pcoulomb3", dispname=f"{str(pcoulomb)}^3",
                        scale=pcoulomb.scale * 3)
ucoulomb2 = Unit.create((ucoulomb ** 2).dim, name="ucoulomb2", dispname=f"{str(ucoulomb)}^2",
                        scale=ucoulomb.scale * 2)
ucoulomb3 = Unit.create((ucoulomb ** 3).dim, name="ucoulomb3", dispname=f"{str(ucoulomb)}^3",
                        scale=ucoulomb.scale * 3)
Tcoulomb2 = Unit.create((Tcoulomb ** 2).dim, name="Tcoulomb2", dispname=f"{str(Tcoulomb)}^2",
                        scale=Tcoulomb.scale * 2)
Tcoulomb3 = Unit.create((Tcoulomb ** 3).dim, name="Tcoulomb3", dispname=f"{str(Tcoulomb)}^3",
                        scale=Tcoulomb.scale * 3)
ycoulomb2 = Unit.create((ycoulomb ** 2).dim, name="ycoulomb2", dispname=f"{str(ycoulomb)}^2",
                        scale=ycoulomb.scale * 2)
ycoulomb3 = Unit.create((ycoulomb ** 3).dim, name="ycoulomb3", dispname=f"{str(ycoulomb)}^3",
                        scale=ycoulomb.scale * 3)
Ecoulomb2 = Unit.create((Ecoulomb ** 2).dim, name="Ecoulomb2", dispname=f"{str(Ecoulomb)}^2",
                        scale=Ecoulomb.scale * 2)
Ecoulomb3 = Unit.create((Ecoulomb ** 3).dim, name="Ecoulomb3", dispname=f"{str(Ecoulomb)}^3",
                        scale=Ecoulomb.scale * 3)
zcoulomb2 = Unit.create((zcoulomb ** 2).dim, name="zcoulomb2", dispname=f"{str(zcoulomb)}^2",
                        scale=zcoulomb.scale * 2)
zcoulomb3 = Unit.create((zcoulomb ** 3).dim, name="zcoulomb3", dispname=f"{str(zcoulomb)}^3",
                        scale=zcoulomb.scale * 3)
Mcoulomb2 = Unit.create((Mcoulomb ** 2).dim, name="Mcoulomb2", dispname=f"{str(Mcoulomb)}^2",
                        scale=Mcoulomb.scale * 2)
Mcoulomb3 = Unit.create((Mcoulomb ** 3).dim, name="Mcoulomb3", dispname=f"{str(Mcoulomb)}^3",
                        scale=Mcoulomb.scale * 3)
kcoulomb2 = Unit.create((kcoulomb ** 2).dim, name="kcoulomb2", dispname=f"{str(kcoulomb)}^2",
                        scale=kcoulomb.scale * 2)
kcoulomb3 = Unit.create((kcoulomb ** 3).dim, name="kcoulomb3", dispname=f"{str(kcoulomb)}^3",
                        scale=kcoulomb.scale * 3)
Ycoulomb2 = Unit.create((Ycoulomb ** 2).dim, name="Ycoulomb2", dispname=f"{str(Ycoulomb)}^2",
                        scale=Ycoulomb.scale * 2)
Ycoulomb3 = Unit.create((Ycoulomb ** 3).dim, name="Ycoulomb3", dispname=f"{str(Ycoulomb)}^3",
                        scale=Ycoulomb.scale * 3)
avolt2 = Unit.create((avolt ** 2).dim, name="avolt2", dispname=f"{str(avolt)}^2", scale=avolt.scale * 2)
avolt3 = Unit.create((avolt ** 3).dim, name="avolt3", dispname=f"{str(avolt)}^3", scale=avolt.scale * 3)
cvolt2 = Unit.create((cvolt ** 2).dim, name="cvolt2", dispname=f"{str(cvolt)}^2", scale=cvolt.scale * 2)
cvolt3 = Unit.create((cvolt ** 3).dim, name="cvolt3", dispname=f"{str(cvolt)}^3", scale=cvolt.scale * 3)
Zvolt2 = Unit.create((Zvolt ** 2).dim, name="Zvolt2", dispname=f"{str(Zvolt)}^2", scale=Zvolt.scale * 2)
Zvolt3 = Unit.create((Zvolt ** 3).dim, name="Zvolt3", dispname=f"{str(Zvolt)}^3", scale=Zvolt.scale * 3)
Pvolt2 = Unit.create((Pvolt ** 2).dim, name="Pvolt2", dispname=f"{str(Pvolt)}^2", scale=Pvolt.scale * 2)
Pvolt3 = Unit.create((Pvolt ** 3).dim, name="Pvolt3", dispname=f"{str(Pvolt)}^3", scale=Pvolt.scale * 3)
dvolt2 = Unit.create((dvolt ** 2).dim, name="dvolt2", dispname=f"{str(dvolt)}^2", scale=dvolt.scale * 2)
dvolt3 = Unit.create((dvolt ** 3).dim, name="dvolt3", dispname=f"{str(dvolt)}^3", scale=dvolt.scale * 3)
Gvolt2 = Unit.create((Gvolt ** 2).dim, name="Gvolt2", dispname=f"{str(Gvolt)}^2", scale=Gvolt.scale * 2)
Gvolt3 = Unit.create((Gvolt ** 3).dim, name="Gvolt3", dispname=f"{str(Gvolt)}^3", scale=Gvolt.scale * 3)
fvolt2 = Unit.create((fvolt ** 2).dim, name="fvolt2", dispname=f"{str(fvolt)}^2", scale=fvolt.scale * 2)
fvolt3 = Unit.create((fvolt ** 3).dim, name="fvolt3", dispname=f"{str(fvolt)}^3", scale=fvolt.scale * 3)
hvolt2 = Unit.create((hvolt ** 2).dim, name="hvolt2", dispname=f"{str(hvolt)}^2", scale=hvolt.scale * 2)
hvolt3 = Unit.create((hvolt ** 3).dim, name="hvolt3", dispname=f"{str(hvolt)}^3", scale=hvolt.scale * 3)
davolt2 = Unit.create((davolt ** 2).dim, name="davolt2", dispname=f"{str(davolt)}^2", scale=davolt.scale * 2)
davolt3 = Unit.create((davolt ** 3).dim, name="davolt3", dispname=f"{str(davolt)}^3", scale=davolt.scale * 3)
mvolt2 = Unit.create((mvolt ** 2).dim, name="mvolt2", dispname=f"{str(mvolt)}^2", scale=mvolt.scale * 2)
mvolt3 = Unit.create((mvolt ** 3).dim, name="mvolt3", dispname=f"{str(mvolt)}^3", scale=mvolt.scale * 3)
nvolt2 = Unit.create((nvolt ** 2).dim, name="nvolt2", dispname=f"{str(nvolt)}^2", scale=nvolt.scale * 2)
nvolt3 = Unit.create((nvolt ** 3).dim, name="nvolt3", dispname=f"{str(nvolt)}^3", scale=nvolt.scale * 3)
pvolt2 = Unit.create((pvolt ** 2).dim, name="pvolt2", dispname=f"{str(pvolt)}^2", scale=pvolt.scale * 2)
pvolt3 = Unit.create((pvolt ** 3).dim, name="pvolt3", dispname=f"{str(pvolt)}^3", scale=pvolt.scale * 3)
uvolt2 = Unit.create((uvolt ** 2).dim, name="uvolt2", dispname=f"{str(uvolt)}^2", scale=uvolt.scale * 2)
uvolt3 = Unit.create((uvolt ** 3).dim, name="uvolt3", dispname=f"{str(uvolt)}^3", scale=uvolt.scale * 3)
Tvolt2 = Unit.create((Tvolt ** 2).dim, name="Tvolt2", dispname=f"{str(Tvolt)}^2", scale=Tvolt.scale * 2)
Tvolt3 = Unit.create((Tvolt ** 3).dim, name="Tvolt3", dispname=f"{str(Tvolt)}^3", scale=Tvolt.scale * 3)
yvolt2 = Unit.create((yvolt ** 2).dim, name="yvolt2", dispname=f"{str(yvolt)}^2", scale=yvolt.scale * 2)
yvolt3 = Unit.create((yvolt ** 3).dim, name="yvolt3", dispname=f"{str(yvolt)}^3", scale=yvolt.scale * 3)
Evolt2 = Unit.create((Evolt ** 2).dim, name="Evolt2", dispname=f"{str(Evolt)}^2", scale=Evolt.scale * 2)
Evolt3 = Unit.create((Evolt ** 3).dim, name="Evolt3", dispname=f"{str(Evolt)}^3", scale=Evolt.scale * 3)
zvolt2 = Unit.create((zvolt ** 2).dim, name="zvolt2", dispname=f"{str(zvolt)}^2", scale=zvolt.scale * 2)
zvolt3 = Unit.create((zvolt ** 3).dim, name="zvolt3", dispname=f"{str(zvolt)}^3", scale=zvolt.scale * 3)
Mvolt2 = Unit.create((Mvolt ** 2).dim, name="Mvolt2", dispname=f"{str(Mvolt)}^2", scale=Mvolt.scale * 2)
Mvolt3 = Unit.create((Mvolt ** 3).dim, name="Mvolt3", dispname=f"{str(Mvolt)}^3", scale=Mvolt.scale * 3)
kvolt2 = Unit.create((kvolt ** 2).dim, name="kvolt2", dispname=f"{str(kvolt)}^2", scale=kvolt.scale * 2)
kvolt3 = Unit.create((kvolt ** 3).dim, name="kvolt3", dispname=f"{str(kvolt)}^3", scale=kvolt.scale * 3)
Yvolt2 = Unit.create((Yvolt ** 2).dim, name="Yvolt2", dispname=f"{str(Yvolt)}^2", scale=Yvolt.scale * 2)
Yvolt3 = Unit.create((Yvolt ** 3).dim, name="Yvolt3", dispname=f"{str(Yvolt)}^3", scale=Yvolt.scale * 3)
afarad2 = Unit.create((afarad ** 2).dim, name="afarad2", dispname=f"{str(afarad)}^2", scale=afarad.scale * 2)
afarad3 = Unit.create((afarad ** 3).dim, name="afarad3", dispname=f"{str(afarad)}^3", scale=afarad.scale * 3)
cfarad2 = Unit.create((cfarad ** 2).dim, name="cfarad2", dispname=f"{str(cfarad)}^2", scale=cfarad.scale * 2)
cfarad3 = Unit.create((cfarad ** 3).dim, name="cfarad3", dispname=f"{str(cfarad)}^3", scale=cfarad.scale * 3)
Zfarad2 = Unit.create((Zfarad ** 2).dim, name="Zfarad2", dispname=f"{str(Zfarad)}^2", scale=Zfarad.scale * 2)
Zfarad3 = Unit.create((Zfarad ** 3).dim, name="Zfarad3", dispname=f"{str(Zfarad)}^3", scale=Zfarad.scale * 3)
Pfarad2 = Unit.create((Pfarad ** 2).dim, name="Pfarad2", dispname=f"{str(Pfarad)}^2", scale=Pfarad.scale * 2)
Pfarad3 = Unit.create((Pfarad ** 3).dim, name="Pfarad3", dispname=f"{str(Pfarad)}^3", scale=Pfarad.scale * 3)
dfarad2 = Unit.create((dfarad ** 2).dim, name="dfarad2", dispname=f"{str(dfarad)}^2", scale=dfarad.scale * 2)
dfarad3 = Unit.create((dfarad ** 3).dim, name="dfarad3", dispname=f"{str(dfarad)}^3", scale=dfarad.scale * 3)
Gfarad2 = Unit.create((Gfarad ** 2).dim, name="Gfarad2", dispname=f"{str(Gfarad)}^2", scale=Gfarad.scale * 2)
Gfarad3 = Unit.create((Gfarad ** 3).dim, name="Gfarad3", dispname=f"{str(Gfarad)}^3", scale=Gfarad.scale * 3)
ffarad2 = Unit.create((ffarad ** 2).dim, name="ffarad2", dispname=f"{str(ffarad)}^2", scale=ffarad.scale * 2)
ffarad3 = Unit.create((ffarad ** 3).dim, name="ffarad3", dispname=f"{str(ffarad)}^3", scale=ffarad.scale * 3)
hfarad2 = Unit.create((hfarad ** 2).dim, name="hfarad2", dispname=f"{str(hfarad)}^2", scale=hfarad.scale * 2)
hfarad3 = Unit.create((hfarad ** 3).dim, name="hfarad3", dispname=f"{str(hfarad)}^3", scale=hfarad.scale * 3)
dafarad2 = Unit.create((dafarad ** 2).dim, name="dafarad2", dispname=f"{str(dafarad)}^2", scale=dafarad.scale * 2)
dafarad3 = Unit.create((dafarad ** 3).dim, name="dafarad3", dispname=f"{str(dafarad)}^3", scale=dafarad.scale * 3)
mfarad2 = Unit.create((mfarad ** 2).dim, name="mfarad2", dispname=f"{str(mfarad)}^2", scale=mfarad.scale * 2)
mfarad3 = Unit.create((mfarad ** 3).dim, name="mfarad3", dispname=f"{str(mfarad)}^3", scale=mfarad.scale * 3)
nfarad2 = Unit.create((nfarad ** 2).dim, name="nfarad2", dispname=f"{str(nfarad)}^2", scale=nfarad.scale * 2)
nfarad3 = Unit.create((nfarad ** 3).dim, name="nfarad3", dispname=f"{str(nfarad)}^3", scale=nfarad.scale * 3)
pfarad2 = Unit.create((pfarad ** 2).dim, name="pfarad2", dispname=f"{str(pfarad)}^2", scale=pfarad.scale * 2)
pfarad3 = Unit.create((pfarad ** 3).dim, name="pfarad3", dispname=f"{str(pfarad)}^3", scale=pfarad.scale * 3)
ufarad2 = Unit.create((ufarad ** 2).dim, name="ufarad2", dispname=f"{str(ufarad)}^2", scale=ufarad.scale * 2)
ufarad3 = Unit.create((ufarad ** 3).dim, name="ufarad3", dispname=f"{str(ufarad)}^3", scale=ufarad.scale * 3)
Tfarad2 = Unit.create((Tfarad ** 2).dim, name="Tfarad2", dispname=f"{str(Tfarad)}^2", scale=Tfarad.scale * 2)
Tfarad3 = Unit.create((Tfarad ** 3).dim, name="Tfarad3", dispname=f"{str(Tfarad)}^3", scale=Tfarad.scale * 3)
yfarad2 = Unit.create((yfarad ** 2).dim, name="yfarad2", dispname=f"{str(yfarad)}^2", scale=yfarad.scale * 2)
yfarad3 = Unit.create((yfarad ** 3).dim, name="yfarad3", dispname=f"{str(yfarad)}^3", scale=yfarad.scale * 3)
Efarad2 = Unit.create((Efarad ** 2).dim, name="Efarad2", dispname=f"{str(Efarad)}^2", scale=Efarad.scale * 2)
Efarad3 = Unit.create((Efarad ** 3).dim, name="Efarad3", dispname=f"{str(Efarad)}^3", scale=Efarad.scale * 3)
zfarad2 = Unit.create((zfarad ** 2).dim, name="zfarad2", dispname=f"{str(zfarad)}^2", scale=zfarad.scale * 2)
zfarad3 = Unit.create((zfarad ** 3).dim, name="zfarad3", dispname=f"{str(zfarad)}^3", scale=zfarad.scale * 3)
Mfarad2 = Unit.create((Mfarad ** 2).dim, name="Mfarad2", dispname=f"{str(Mfarad)}^2", scale=Mfarad.scale * 2)
Mfarad3 = Unit.create((Mfarad ** 3).dim, name="Mfarad3", dispname=f"{str(Mfarad)}^3", scale=Mfarad.scale * 3)
kfarad2 = Unit.create((kfarad ** 2).dim, name="kfarad2", dispname=f"{str(kfarad)}^2", scale=kfarad.scale * 2)
kfarad3 = Unit.create((kfarad ** 3).dim, name="kfarad3", dispname=f"{str(kfarad)}^3", scale=kfarad.scale * 3)
Yfarad2 = Unit.create((Yfarad ** 2).dim, name="Yfarad2", dispname=f"{str(Yfarad)}^2", scale=Yfarad.scale * 2)
Yfarad3 = Unit.create((Yfarad ** 3).dim, name="Yfarad3", dispname=f"{str(Yfarad)}^3", scale=Yfarad.scale * 3)
aohm2 = Unit.create((aohm ** 2).dim, name="aohm2", dispname=f"{str(aohm)}^2", scale=aohm.scale * 2)
aohm3 = Unit.create((aohm ** 3).dim, name="aohm3", dispname=f"{str(aohm)}^3", scale=aohm.scale * 3)
cohm2 = Unit.create((cohm ** 2).dim, name="cohm2", dispname=f"{str(cohm)}^2", scale=cohm.scale * 2)
cohm3 = Unit.create((cohm ** 3).dim, name="cohm3", dispname=f"{str(cohm)}^3", scale=cohm.scale * 3)
Zohm2 = Unit.create((Zohm ** 2).dim, name="Zohm2", dispname=f"{str(Zohm)}^2", scale=Zohm.scale * 2)
Zohm3 = Unit.create((Zohm ** 3).dim, name="Zohm3", dispname=f"{str(Zohm)}^3", scale=Zohm.scale * 3)
Pohm2 = Unit.create((Pohm ** 2).dim, name="Pohm2", dispname=f"{str(Pohm)}^2", scale=Pohm.scale * 2)
Pohm3 = Unit.create((Pohm ** 3).dim, name="Pohm3", dispname=f"{str(Pohm)}^3", scale=Pohm.scale * 3)
dohm2 = Unit.create((dohm ** 2).dim, name="dohm2", dispname=f"{str(dohm)}^2", scale=dohm.scale * 2)
dohm3 = Unit.create((dohm ** 3).dim, name="dohm3", dispname=f"{str(dohm)}^3", scale=dohm.scale * 3)
Gohm2 = Unit.create((Gohm ** 2).dim, name="Gohm2", dispname=f"{str(Gohm)}^2", scale=Gohm.scale * 2)
Gohm3 = Unit.create((Gohm ** 3).dim, name="Gohm3", dispname=f"{str(Gohm)}^3", scale=Gohm.scale * 3)
fohm2 = Unit.create((fohm ** 2).dim, name="fohm2", dispname=f"{str(fohm)}^2", scale=fohm.scale * 2)
fohm3 = Unit.create((fohm ** 3).dim, name="fohm3", dispname=f"{str(fohm)}^3", scale=fohm.scale * 3)
hohm2 = Unit.create((hohm ** 2).dim, name="hohm2", dispname=f"{str(hohm)}^2", scale=hohm.scale * 2)
hohm3 = Unit.create((hohm ** 3).dim, name="hohm3", dispname=f"{str(hohm)}^3", scale=hohm.scale * 3)
daohm2 = Unit.create((daohm ** 2).dim, name="daohm2", dispname=f"{str(daohm)}^2", scale=daohm.scale * 2)
daohm3 = Unit.create((daohm ** 3).dim, name="daohm3", dispname=f"{str(daohm)}^3", scale=daohm.scale * 3)
mohm2 = Unit.create((mohm ** 2).dim, name="mohm2", dispname=f"{str(mohm)}^2", scale=mohm.scale * 2)
mohm3 = Unit.create((mohm ** 3).dim, name="mohm3", dispname=f"{str(mohm)}^3", scale=mohm.scale * 3)
nohm2 = Unit.create((nohm ** 2).dim, name="nohm2", dispname=f"{str(nohm)}^2", scale=nohm.scale * 2)
nohm3 = Unit.create((nohm ** 3).dim, name="nohm3", dispname=f"{str(nohm)}^3", scale=nohm.scale * 3)
pohm2 = Unit.create((pohm ** 2).dim, name="pohm2", dispname=f"{str(pohm)}^2", scale=pohm.scale * 2)
pohm3 = Unit.create((pohm ** 3).dim, name="pohm3", dispname=f"{str(pohm)}^3", scale=pohm.scale * 3)
uohm2 = Unit.create((uohm ** 2).dim, name="uohm2", dispname=f"{str(uohm)}^2", scale=uohm.scale * 2)
uohm3 = Unit.create((uohm ** 3).dim, name="uohm3", dispname=f"{str(uohm)}^3", scale=uohm.scale * 3)
Tohm2 = Unit.create((Tohm ** 2).dim, name="Tohm2", dispname=f"{str(Tohm)}^2", scale=Tohm.scale * 2)
Tohm3 = Unit.create((Tohm ** 3).dim, name="Tohm3", dispname=f"{str(Tohm)}^3", scale=Tohm.scale * 3)
yohm2 = Unit.create((yohm ** 2).dim, name="yohm2", dispname=f"{str(yohm)}^2", scale=yohm.scale * 2)
yohm3 = Unit.create((yohm ** 3).dim, name="yohm3", dispname=f"{str(yohm)}^3", scale=yohm.scale * 3)
Eohm2 = Unit.create((Eohm ** 2).dim, name="Eohm2", dispname=f"{str(Eohm)}^2", scale=Eohm.scale * 2)
Eohm3 = Unit.create((Eohm ** 3).dim, name="Eohm3", dispname=f"{str(Eohm)}^3", scale=Eohm.scale * 3)
zohm2 = Unit.create((zohm ** 2).dim, name="zohm2", dispname=f"{str(zohm)}^2", scale=zohm.scale * 2)
zohm3 = Unit.create((zohm ** 3).dim, name="zohm3", dispname=f"{str(zohm)}^3", scale=zohm.scale * 3)
Mohm2 = Unit.create((Mohm ** 2).dim, name="Mohm2", dispname=f"{str(Mohm)}^2", scale=Mohm.scale * 2)
Mohm3 = Unit.create((Mohm ** 3).dim, name="Mohm3", dispname=f"{str(Mohm)}^3", scale=Mohm.scale * 3)
kohm2 = Unit.create((kohm ** 2).dim, name="kohm2", dispname=f"{str(kohm)}^2", scale=kohm.scale * 2)
kohm3 = Unit.create((kohm ** 3).dim, name="kohm3", dispname=f"{str(kohm)}^3", scale=kohm.scale * 3)
Yohm2 = Unit.create((Yohm ** 2).dim, name="Yohm2", dispname=f"{str(Yohm)}^2", scale=Yohm.scale * 2)
Yohm3 = Unit.create((Yohm ** 3).dim, name="Yohm3", dispname=f"{str(Yohm)}^3", scale=Yohm.scale * 3)
asiemens2 = Unit.create((asiemens ** 2).dim, name="asiemens2", dispname=f"{str(asiemens)}^2",
                        scale=asiemens.scale * 2)
asiemens3 = Unit.create((asiemens ** 3).dim, name="asiemens3", dispname=f"{str(asiemens)}^3",
                        scale=asiemens.scale * 3)
csiemens2 = Unit.create((csiemens ** 2).dim, name="csiemens2", dispname=f"{str(csiemens)}^2",
                        scale=csiemens.scale * 2)
csiemens3 = Unit.create((csiemens ** 3).dim, name="csiemens3", dispname=f"{str(csiemens)}^3",
                        scale=csiemens.scale * 3)
Zsiemens2 = Unit.create((Zsiemens ** 2).dim, name="Zsiemens2", dispname=f"{str(Zsiemens)}^2",
                        scale=Zsiemens.scale * 2)
Zsiemens3 = Unit.create((Zsiemens ** 3).dim, name="Zsiemens3", dispname=f"{str(Zsiemens)}^3",
                        scale=Zsiemens.scale * 3)
Psiemens2 = Unit.create((Psiemens ** 2).dim, name="Psiemens2", dispname=f"{str(Psiemens)}^2",
                        scale=Psiemens.scale * 2)
Psiemens3 = Unit.create((Psiemens ** 3).dim, name="Psiemens3", dispname=f"{str(Psiemens)}^3",
                        scale=Psiemens.scale * 3)
dsiemens2 = Unit.create((dsiemens ** 2).dim, name="dsiemens2", dispname=f"{str(dsiemens)}^2",
                        scale=dsiemens.scale * 2)
dsiemens3 = Unit.create((dsiemens ** 3).dim, name="dsiemens3", dispname=f"{str(dsiemens)}^3",
                        scale=dsiemens.scale * 3)
Gsiemens2 = Unit.create((Gsiemens ** 2).dim, name="Gsiemens2", dispname=f"{str(Gsiemens)}^2",
                        scale=Gsiemens.scale * 2)
Gsiemens3 = Unit.create((Gsiemens ** 3).dim, name="Gsiemens3", dispname=f"{str(Gsiemens)}^3",
                        scale=Gsiemens.scale * 3)
fsiemens2 = Unit.create((fsiemens ** 2).dim, name="fsiemens2", dispname=f"{str(fsiemens)}^2",
                        scale=fsiemens.scale * 2)
fsiemens3 = Unit.create((fsiemens ** 3).dim, name="fsiemens3", dispname=f"{str(fsiemens)}^3",
                        scale=fsiemens.scale * 3)
hsiemens2 = Unit.create((hsiemens ** 2).dim, name="hsiemens2", dispname=f"{str(hsiemens)}^2",
                        scale=hsiemens.scale * 2)
hsiemens3 = Unit.create((hsiemens ** 3).dim, name="hsiemens3", dispname=f"{str(hsiemens)}^3",
                        scale=hsiemens.scale * 3)
dasiemens2 = Unit.create((dasiemens ** 2).dim, name="dasiemens2", dispname=f"{str(dasiemens)}^2",
                         scale=dasiemens.scale * 2)
dasiemens3 = Unit.create((dasiemens ** 3).dim, name="dasiemens3", dispname=f"{str(dasiemens)}^3",
                         scale=dasiemens.scale * 3)
msiemens2 = Unit.create((msiemens ** 2).dim, name="msiemens2", dispname=f"{str(msiemens)}^2",
                        scale=msiemens.scale * 2)
msiemens3 = Unit.create((msiemens ** 3).dim, name="msiemens3", dispname=f"{str(msiemens)}^3",
                        scale=msiemens.scale * 3)
nsiemens2 = Unit.create((nsiemens ** 2).dim, name="nsiemens2", dispname=f"{str(nsiemens)}^2",
                        scale=nsiemens.scale * 2)
nsiemens3 = Unit.create((nsiemens ** 3).dim, name="nsiemens3", dispname=f"{str(nsiemens)}^3",
                        scale=nsiemens.scale * 3)
psiemens2 = Unit.create((psiemens ** 2).dim, name="psiemens2", dispname=f"{str(psiemens)}^2",
                        scale=psiemens.scale * 2)
psiemens3 = Unit.create((psiemens ** 3).dim, name="psiemens3", dispname=f"{str(psiemens)}^3",
                        scale=psiemens.scale * 3)
usiemens2 = Unit.create((usiemens ** 2).dim, name="usiemens2", dispname=f"{str(usiemens)}^2",
                        scale=usiemens.scale * 2)
usiemens3 = Unit.create((usiemens ** 3).dim, name="usiemens3", dispname=f"{str(usiemens)}^3",
                        scale=usiemens.scale * 3)
Tsiemens2 = Unit.create((Tsiemens ** 2).dim, name="Tsiemens2", dispname=f"{str(Tsiemens)}^2",
                        scale=Tsiemens.scale * 2)
Tsiemens3 = Unit.create((Tsiemens ** 3).dim, name="Tsiemens3", dispname=f"{str(Tsiemens)}^3",
                        scale=Tsiemens.scale * 3)
ysiemens2 = Unit.create((ysiemens ** 2).dim, name="ysiemens2", dispname=f"{str(ysiemens)}^2",
                        scale=ysiemens.scale * 2)
ysiemens3 = Unit.create((ysiemens ** 3).dim, name="ysiemens3", dispname=f"{str(ysiemens)}^3",
                        scale=ysiemens.scale * 3)
Esiemens2 = Unit.create((Esiemens ** 2).dim, name="Esiemens2", dispname=f"{str(Esiemens)}^2",
                        scale=Esiemens.scale * 2)
Esiemens3 = Unit.create((Esiemens ** 3).dim, name="Esiemens3", dispname=f"{str(Esiemens)}^3",
                        scale=Esiemens.scale * 3)
zsiemens2 = Unit.create((zsiemens ** 2).dim, name="zsiemens2", dispname=f"{str(zsiemens)}^2",
                        scale=zsiemens.scale * 2)
zsiemens3 = Unit.create((zsiemens ** 3).dim, name="zsiemens3", dispname=f"{str(zsiemens)}^3",
                        scale=zsiemens.scale * 3)
Msiemens2 = Unit.create((Msiemens ** 2).dim, name="Msiemens2", dispname=f"{str(Msiemens)}^2",
                        scale=Msiemens.scale * 2)
Msiemens3 = Unit.create((Msiemens ** 3).dim, name="Msiemens3", dispname=f"{str(Msiemens)}^3",
                        scale=Msiemens.scale * 3)
ksiemens2 = Unit.create((ksiemens ** 2).dim, name="ksiemens2", dispname=f"{str(ksiemens)}^2",
                        scale=ksiemens.scale * 2)
ksiemens3 = Unit.create((ksiemens ** 3).dim, name="ksiemens3", dispname=f"{str(ksiemens)}^3",
                        scale=ksiemens.scale * 3)
Ysiemens2 = Unit.create((Ysiemens ** 2).dim, name="Ysiemens2", dispname=f"{str(Ysiemens)}^2",
                        scale=Ysiemens.scale * 2)
Ysiemens3 = Unit.create((Ysiemens ** 3).dim, name="Ysiemens3", dispname=f"{str(Ysiemens)}^3",
                        scale=Ysiemens.scale * 3)
aweber2 = Unit.create((aweber ** 2).dim, name="aweber2", dispname=f"{str(aweber)}^2", scale=aweber.scale * 2)
aweber3 = Unit.create((aweber ** 3).dim, name="aweber3", dispname=f"{str(aweber)}^3", scale=aweber.scale * 3)
cweber2 = Unit.create((cweber ** 2).dim, name="cweber2", dispname=f"{str(cweber)}^2", scale=cweber.scale * 2)
cweber3 = Unit.create((cweber ** 3).dim, name="cweber3", dispname=f"{str(cweber)}^3", scale=cweber.scale * 3)
Zweber2 = Unit.create((Zweber ** 2).dim, name="Zweber2", dispname=f"{str(Zweber)}^2", scale=Zweber.scale * 2)
Zweber3 = Unit.create((Zweber ** 3).dim, name="Zweber3", dispname=f"{str(Zweber)}^3", scale=Zweber.scale * 3)
Pweber2 = Unit.create((Pweber ** 2).dim, name="Pweber2", dispname=f"{str(Pweber)}^2", scale=Pweber.scale * 2)
Pweber3 = Unit.create((Pweber ** 3).dim, name="Pweber3", dispname=f"{str(Pweber)}^3", scale=Pweber.scale * 3)
dweber2 = Unit.create((dweber ** 2).dim, name="dweber2", dispname=f"{str(dweber)}^2", scale=dweber.scale * 2)
dweber3 = Unit.create((dweber ** 3).dim, name="dweber3", dispname=f"{str(dweber)}^3", scale=dweber.scale * 3)
Gweber2 = Unit.create((Gweber ** 2).dim, name="Gweber2", dispname=f"{str(Gweber)}^2", scale=Gweber.scale * 2)
Gweber3 = Unit.create((Gweber ** 3).dim, name="Gweber3", dispname=f"{str(Gweber)}^3", scale=Gweber.scale * 3)
fweber2 = Unit.create((fweber ** 2).dim, name="fweber2", dispname=f"{str(fweber)}^2", scale=fweber.scale * 2)
fweber3 = Unit.create((fweber ** 3).dim, name="fweber3", dispname=f"{str(fweber)}^3", scale=fweber.scale * 3)
hweber2 = Unit.create((hweber ** 2).dim, name="hweber2", dispname=f"{str(hweber)}^2", scale=hweber.scale * 2)
hweber3 = Unit.create((hweber ** 3).dim, name="hweber3", dispname=f"{str(hweber)}^3", scale=hweber.scale * 3)
daweber2 = Unit.create((daweber ** 2).dim, name="daweber2", dispname=f"{str(daweber)}^2", scale=daweber.scale * 2)
daweber3 = Unit.create((daweber ** 3).dim, name="daweber3", dispname=f"{str(daweber)}^3", scale=daweber.scale * 3)
mweber2 = Unit.create((mweber ** 2).dim, name="mweber2", dispname=f"{str(mweber)}^2", scale=mweber.scale * 2)
mweber3 = Unit.create((mweber ** 3).dim, name="mweber3", dispname=f"{str(mweber)}^3", scale=mweber.scale * 3)
nweber2 = Unit.create((nweber ** 2).dim, name="nweber2", dispname=f"{str(nweber)}^2", scale=nweber.scale * 2)
nweber3 = Unit.create((nweber ** 3).dim, name="nweber3", dispname=f"{str(nweber)}^3", scale=nweber.scale * 3)
pweber2 = Unit.create((pweber ** 2).dim, name="pweber2", dispname=f"{str(pweber)}^2", scale=pweber.scale * 2)
pweber3 = Unit.create((pweber ** 3).dim, name="pweber3", dispname=f"{str(pweber)}^3", scale=pweber.scale * 3)
uweber2 = Unit.create((uweber ** 2).dim, name="uweber2", dispname=f"{str(uweber)}^2", scale=uweber.scale * 2)
uweber3 = Unit.create((uweber ** 3).dim, name="uweber3", dispname=f"{str(uweber)}^3", scale=uweber.scale * 3)
Tweber2 = Unit.create((Tweber ** 2).dim, name="Tweber2", dispname=f"{str(Tweber)}^2", scale=Tweber.scale * 2)
Tweber3 = Unit.create((Tweber ** 3).dim, name="Tweber3", dispname=f"{str(Tweber)}^3", scale=Tweber.scale * 3)
yweber2 = Unit.create((yweber ** 2).dim, name="yweber2", dispname=f"{str(yweber)}^2", scale=yweber.scale * 2)
yweber3 = Unit.create((yweber ** 3).dim, name="yweber3", dispname=f"{str(yweber)}^3", scale=yweber.scale * 3)
Eweber2 = Unit.create((Eweber ** 2).dim, name="Eweber2", dispname=f"{str(Eweber)}^2", scale=Eweber.scale * 2)
Eweber3 = Unit.create((Eweber ** 3).dim, name="Eweber3", dispname=f"{str(Eweber)}^3", scale=Eweber.scale * 3)
zweber2 = Unit.create((zweber ** 2).dim, name="zweber2", dispname=f"{str(zweber)}^2", scale=zweber.scale * 2)
zweber3 = Unit.create((zweber ** 3).dim, name="zweber3", dispname=f"{str(zweber)}^3", scale=zweber.scale * 3)
Mweber2 = Unit.create((Mweber ** 2).dim, name="Mweber2", dispname=f"{str(Mweber)}^2", scale=Mweber.scale * 2)
Mweber3 = Unit.create((Mweber ** 3).dim, name="Mweber3", dispname=f"{str(Mweber)}^3", scale=Mweber.scale * 3)
kweber2 = Unit.create((kweber ** 2).dim, name="kweber2", dispname=f"{str(kweber)}^2", scale=kweber.scale * 2)
kweber3 = Unit.create((kweber ** 3).dim, name="kweber3", dispname=f"{str(kweber)}^3", scale=kweber.scale * 3)
Yweber2 = Unit.create((Yweber ** 2).dim, name="Yweber2", dispname=f"{str(Yweber)}^2", scale=Yweber.scale * 2)
Yweber3 = Unit.create((Yweber ** 3).dim, name="Yweber3", dispname=f"{str(Yweber)}^3", scale=Yweber.scale * 3)
atesla2 = Unit.create((atesla ** 2).dim, name="atesla2", dispname=f"{str(atesla)}^2", scale=atesla.scale * 2)
atesla3 = Unit.create((atesla ** 3).dim, name="atesla3", dispname=f"{str(atesla)}^3", scale=atesla.scale * 3)
ctesla2 = Unit.create((ctesla ** 2).dim, name="ctesla2", dispname=f"{str(ctesla)}^2", scale=ctesla.scale * 2)
ctesla3 = Unit.create((ctesla ** 3).dim, name="ctesla3", dispname=f"{str(ctesla)}^3", scale=ctesla.scale * 3)
Ztesla2 = Unit.create((Ztesla ** 2).dim, name="Ztesla2", dispname=f"{str(Ztesla)}^2", scale=Ztesla.scale * 2)
Ztesla3 = Unit.create((Ztesla ** 3).dim, name="Ztesla3", dispname=f"{str(Ztesla)}^3", scale=Ztesla.scale * 3)
Ptesla2 = Unit.create((Ptesla ** 2).dim, name="Ptesla2", dispname=f"{str(Ptesla)}^2", scale=Ptesla.scale * 2)
Ptesla3 = Unit.create((Ptesla ** 3).dim, name="Ptesla3", dispname=f"{str(Ptesla)}^3", scale=Ptesla.scale * 3)
dtesla2 = Unit.create((dtesla ** 2).dim, name="dtesla2", dispname=f"{str(dtesla)}^2", scale=dtesla.scale * 2)
dtesla3 = Unit.create((dtesla ** 3).dim, name="dtesla3", dispname=f"{str(dtesla)}^3", scale=dtesla.scale * 3)
Gtesla2 = Unit.create((Gtesla ** 2).dim, name="Gtesla2", dispname=f"{str(Gtesla)}^2", scale=Gtesla.scale * 2)
Gtesla3 = Unit.create((Gtesla ** 3).dim, name="Gtesla3", dispname=f"{str(Gtesla)}^3", scale=Gtesla.scale * 3)
ftesla2 = Unit.create((ftesla ** 2).dim, name="ftesla2", dispname=f"{str(ftesla)}^2", scale=ftesla.scale * 2)
ftesla3 = Unit.create((ftesla ** 3).dim, name="ftesla3", dispname=f"{str(ftesla)}^3", scale=ftesla.scale * 3)
htesla2 = Unit.create((htesla ** 2).dim, name="htesla2", dispname=f"{str(htesla)}^2", scale=htesla.scale * 2)
htesla3 = Unit.create((htesla ** 3).dim, name="htesla3", dispname=f"{str(htesla)}^3", scale=htesla.scale * 3)
datesla2 = Unit.create((datesla ** 2).dim, name="datesla2", dispname=f"{str(datesla)}^2", scale=datesla.scale * 2)
datesla3 = Unit.create((datesla ** 3).dim, name="datesla3", dispname=f"{str(datesla)}^3", scale=datesla.scale * 3)
mtesla2 = Unit.create((mtesla ** 2).dim, name="mtesla2", dispname=f"{str(mtesla)}^2", scale=mtesla.scale * 2)
mtesla3 = Unit.create((mtesla ** 3).dim, name="mtesla3", dispname=f"{str(mtesla)}^3", scale=mtesla.scale * 3)
ntesla2 = Unit.create((ntesla ** 2).dim, name="ntesla2", dispname=f"{str(ntesla)}^2", scale=ntesla.scale * 2)
ntesla3 = Unit.create((ntesla ** 3).dim, name="ntesla3", dispname=f"{str(ntesla)}^3", scale=ntesla.scale * 3)
ptesla2 = Unit.create((ptesla ** 2).dim, name="ptesla2", dispname=f"{str(ptesla)}^2", scale=ptesla.scale * 2)
ptesla3 = Unit.create((ptesla ** 3).dim, name="ptesla3", dispname=f"{str(ptesla)}^3", scale=ptesla.scale * 3)
utesla2 = Unit.create((utesla ** 2).dim, name="utesla2", dispname=f"{str(utesla)}^2", scale=utesla.scale * 2)
utesla3 = Unit.create((utesla ** 3).dim, name="utesla3", dispname=f"{str(utesla)}^3", scale=utesla.scale * 3)
Ttesla2 = Unit.create((Ttesla ** 2).dim, name="Ttesla2", dispname=f"{str(Ttesla)}^2", scale=Ttesla.scale * 2)
Ttesla3 = Unit.create((Ttesla ** 3).dim, name="Ttesla3", dispname=f"{str(Ttesla)}^3", scale=Ttesla.scale * 3)
ytesla2 = Unit.create((ytesla ** 2).dim, name="ytesla2", dispname=f"{str(ytesla)}^2", scale=ytesla.scale * 2)
ytesla3 = Unit.create((ytesla ** 3).dim, name="ytesla3", dispname=f"{str(ytesla)}^3", scale=ytesla.scale * 3)
Etesla2 = Unit.create((Etesla ** 2).dim, name="Etesla2", dispname=f"{str(Etesla)}^2", scale=Etesla.scale * 2)
Etesla3 = Unit.create((Etesla ** 3).dim, name="Etesla3", dispname=f"{str(Etesla)}^3", scale=Etesla.scale * 3)
ztesla2 = Unit.create((ztesla ** 2).dim, name="ztesla2", dispname=f"{str(ztesla)}^2", scale=ztesla.scale * 2)
ztesla3 = Unit.create((ztesla ** 3).dim, name="ztesla3", dispname=f"{str(ztesla)}^3", scale=ztesla.scale * 3)
Mtesla2 = Unit.create((Mtesla ** 2).dim, name="Mtesla2", dispname=f"{str(Mtesla)}^2", scale=Mtesla.scale * 2)
Mtesla3 = Unit.create((Mtesla ** 3).dim, name="Mtesla3", dispname=f"{str(Mtesla)}^3", scale=Mtesla.scale * 3)
ktesla2 = Unit.create((ktesla ** 2).dim, name="ktesla2", dispname=f"{str(ktesla)}^2", scale=ktesla.scale * 2)
ktesla3 = Unit.create((ktesla ** 3).dim, name="ktesla3", dispname=f"{str(ktesla)}^3", scale=ktesla.scale * 3)
Ytesla2 = Unit.create((Ytesla ** 2).dim, name="Ytesla2", dispname=f"{str(Ytesla)}^2", scale=Ytesla.scale * 2)
Ytesla3 = Unit.create((Ytesla ** 3).dim, name="Ytesla3", dispname=f"{str(Ytesla)}^3", scale=Ytesla.scale * 3)
ahenry2 = Unit.create((ahenry ** 2).dim, name="ahenry2", dispname=f"{str(ahenry)}^2", scale=ahenry.scale * 2)
ahenry3 = Unit.create((ahenry ** 3).dim, name="ahenry3", dispname=f"{str(ahenry)}^3", scale=ahenry.scale * 3)
chenry2 = Unit.create((chenry ** 2).dim, name="chenry2", dispname=f"{str(chenry)}^2", scale=chenry.scale * 2)
chenry3 = Unit.create((chenry ** 3).dim, name="chenry3", dispname=f"{str(chenry)}^3", scale=chenry.scale * 3)
Zhenry2 = Unit.create((Zhenry ** 2).dim, name="Zhenry2", dispname=f"{str(Zhenry)}^2", scale=Zhenry.scale * 2)
Zhenry3 = Unit.create((Zhenry ** 3).dim, name="Zhenry3", dispname=f"{str(Zhenry)}^3", scale=Zhenry.scale * 3)
Phenry2 = Unit.create((Phenry ** 2).dim, name="Phenry2", dispname=f"{str(Phenry)}^2", scale=Phenry.scale * 2)
Phenry3 = Unit.create((Phenry ** 3).dim, name="Phenry3", dispname=f"{str(Phenry)}^3", scale=Phenry.scale * 3)
dhenry2 = Unit.create((dhenry ** 2).dim, name="dhenry2", dispname=f"{str(dhenry)}^2", scale=dhenry.scale * 2)
dhenry3 = Unit.create((dhenry ** 3).dim, name="dhenry3", dispname=f"{str(dhenry)}^3", scale=dhenry.scale * 3)
Ghenry2 = Unit.create((Ghenry ** 2).dim, name="Ghenry2", dispname=f"{str(Ghenry)}^2", scale=Ghenry.scale * 2)
Ghenry3 = Unit.create((Ghenry ** 3).dim, name="Ghenry3", dispname=f"{str(Ghenry)}^3", scale=Ghenry.scale * 3)
fhenry2 = Unit.create((fhenry ** 2).dim, name="fhenry2", dispname=f"{str(fhenry)}^2", scale=fhenry.scale * 2)
fhenry3 = Unit.create((fhenry ** 3).dim, name="fhenry3", dispname=f"{str(fhenry)}^3", scale=fhenry.scale * 3)
hhenry2 = Unit.create((hhenry ** 2).dim, name="hhenry2", dispname=f"{str(hhenry)}^2", scale=hhenry.scale * 2)
hhenry3 = Unit.create((hhenry ** 3).dim, name="hhenry3", dispname=f"{str(hhenry)}^3", scale=hhenry.scale * 3)
dahenry2 = Unit.create((dahenry ** 2).dim, name="dahenry2", dispname=f"{str(dahenry)}^2", scale=dahenry.scale * 2)
dahenry3 = Unit.create((dahenry ** 3).dim, name="dahenry3", dispname=f"{str(dahenry)}^3", scale=dahenry.scale * 3)
mhenry2 = Unit.create((mhenry ** 2).dim, name="mhenry2", dispname=f"{str(mhenry)}^2", scale=mhenry.scale * 2)
mhenry3 = Unit.create((mhenry ** 3).dim, name="mhenry3", dispname=f"{str(mhenry)}^3", scale=mhenry.scale * 3)
nhenry2 = Unit.create((nhenry ** 2).dim, name="nhenry2", dispname=f"{str(nhenry)}^2", scale=nhenry.scale * 2)
nhenry3 = Unit.create((nhenry ** 3).dim, name="nhenry3", dispname=f"{str(nhenry)}^3", scale=nhenry.scale * 3)
phenry2 = Unit.create((phenry ** 2).dim, name="phenry2", dispname=f"{str(phenry)}^2", scale=phenry.scale * 2)
phenry3 = Unit.create((phenry ** 3).dim, name="phenry3", dispname=f"{str(phenry)}^3", scale=phenry.scale * 3)
uhenry2 = Unit.create((uhenry ** 2).dim, name="uhenry2", dispname=f"{str(uhenry)}^2", scale=uhenry.scale * 2)
uhenry3 = Unit.create((uhenry ** 3).dim, name="uhenry3", dispname=f"{str(uhenry)}^3", scale=uhenry.scale * 3)
Thenry2 = Unit.create((Thenry ** 2).dim, name="Thenry2", dispname=f"{str(Thenry)}^2", scale=Thenry.scale * 2)
Thenry3 = Unit.create((Thenry ** 3).dim, name="Thenry3", dispname=f"{str(Thenry)}^3", scale=Thenry.scale * 3)
yhenry2 = Unit.create((yhenry ** 2).dim, name="yhenry2", dispname=f"{str(yhenry)}^2", scale=yhenry.scale * 2)
yhenry3 = Unit.create((yhenry ** 3).dim, name="yhenry3", dispname=f"{str(yhenry)}^3", scale=yhenry.scale * 3)
Ehenry2 = Unit.create((Ehenry ** 2).dim, name="Ehenry2", dispname=f"{str(Ehenry)}^2", scale=Ehenry.scale * 2)
Ehenry3 = Unit.create((Ehenry ** 3).dim, name="Ehenry3", dispname=f"{str(Ehenry)}^3", scale=Ehenry.scale * 3)
zhenry2 = Unit.create((zhenry ** 2).dim, name="zhenry2", dispname=f"{str(zhenry)}^2", scale=zhenry.scale * 2)
zhenry3 = Unit.create((zhenry ** 3).dim, name="zhenry3", dispname=f"{str(zhenry)}^3", scale=zhenry.scale * 3)
Mhenry2 = Unit.create((Mhenry ** 2).dim, name="Mhenry2", dispname=f"{str(Mhenry)}^2", scale=Mhenry.scale * 2)
Mhenry3 = Unit.create((Mhenry ** 3).dim, name="Mhenry3", dispname=f"{str(Mhenry)}^3", scale=Mhenry.scale * 3)
khenry2 = Unit.create((khenry ** 2).dim, name="khenry2", dispname=f"{str(khenry)}^2", scale=khenry.scale * 2)
khenry3 = Unit.create((khenry ** 3).dim, name="khenry3", dispname=f"{str(khenry)}^3", scale=khenry.scale * 3)
Yhenry2 = Unit.create((Yhenry ** 2).dim, name="Yhenry2", dispname=f"{str(Yhenry)}^2", scale=Yhenry.scale * 2)
Yhenry3 = Unit.create((Yhenry ** 3).dim, name="Yhenry3", dispname=f"{str(Yhenry)}^3", scale=Yhenry.scale * 3)
alumen2 = Unit.create((alumen ** 2).dim, name="alumen2", dispname=f"{str(alumen)}^2", scale=alumen.scale * 2)
alumen3 = Unit.create((alumen ** 3).dim, name="alumen3", dispname=f"{str(alumen)}^3", scale=alumen.scale * 3)
clumen2 = Unit.create((clumen ** 2).dim, name="clumen2", dispname=f"{str(clumen)}^2", scale=clumen.scale * 2)
clumen3 = Unit.create((clumen ** 3).dim, name="clumen3", dispname=f"{str(clumen)}^3", scale=clumen.scale * 3)
Zlumen2 = Unit.create((Zlumen ** 2).dim, name="Zlumen2", dispname=f"{str(Zlumen)}^2", scale=Zlumen.scale * 2)
Zlumen3 = Unit.create((Zlumen ** 3).dim, name="Zlumen3", dispname=f"{str(Zlumen)}^3", scale=Zlumen.scale * 3)
Plumen2 = Unit.create((Plumen ** 2).dim, name="Plumen2", dispname=f"{str(Plumen)}^2", scale=Plumen.scale * 2)
Plumen3 = Unit.create((Plumen ** 3).dim, name="Plumen3", dispname=f"{str(Plumen)}^3", scale=Plumen.scale * 3)
dlumen2 = Unit.create((dlumen ** 2).dim, name="dlumen2", dispname=f"{str(dlumen)}^2", scale=dlumen.scale * 2)
dlumen3 = Unit.create((dlumen ** 3).dim, name="dlumen3", dispname=f"{str(dlumen)}^3", scale=dlumen.scale * 3)
Glumen2 = Unit.create((Glumen ** 2).dim, name="Glumen2", dispname=f"{str(Glumen)}^2", scale=Glumen.scale * 2)
Glumen3 = Unit.create((Glumen ** 3).dim, name="Glumen3", dispname=f"{str(Glumen)}^3", scale=Glumen.scale * 3)
flumen2 = Unit.create((flumen ** 2).dim, name="flumen2", dispname=f"{str(flumen)}^2", scale=flumen.scale * 2)
flumen3 = Unit.create((flumen ** 3).dim, name="flumen3", dispname=f"{str(flumen)}^3", scale=flumen.scale * 3)
hlumen2 = Unit.create((hlumen ** 2).dim, name="hlumen2", dispname=f"{str(hlumen)}^2", scale=hlumen.scale * 2)
hlumen3 = Unit.create((hlumen ** 3).dim, name="hlumen3", dispname=f"{str(hlumen)}^3", scale=hlumen.scale * 3)
dalumen2 = Unit.create((dalumen ** 2).dim, name="dalumen2", dispname=f"{str(dalumen)}^2", scale=dalumen.scale * 2)
dalumen3 = Unit.create((dalumen ** 3).dim, name="dalumen3", dispname=f"{str(dalumen)}^3", scale=dalumen.scale * 3)
mlumen2 = Unit.create((mlumen ** 2).dim, name="mlumen2", dispname=f"{str(mlumen)}^2", scale=mlumen.scale * 2)
mlumen3 = Unit.create((mlumen ** 3).dim, name="mlumen3", dispname=f"{str(mlumen)}^3", scale=mlumen.scale * 3)
nlumen2 = Unit.create((nlumen ** 2).dim, name="nlumen2", dispname=f"{str(nlumen)}^2", scale=nlumen.scale * 2)
nlumen3 = Unit.create((nlumen ** 3).dim, name="nlumen3", dispname=f"{str(nlumen)}^3", scale=nlumen.scale * 3)
plumen2 = Unit.create((plumen ** 2).dim, name="plumen2", dispname=f"{str(plumen)}^2", scale=plumen.scale * 2)
plumen3 = Unit.create((plumen ** 3).dim, name="plumen3", dispname=f"{str(plumen)}^3", scale=plumen.scale * 3)
ulumen2 = Unit.create((ulumen ** 2).dim, name="ulumen2", dispname=f"{str(ulumen)}^2", scale=ulumen.scale * 2)
ulumen3 = Unit.create((ulumen ** 3).dim, name="ulumen3", dispname=f"{str(ulumen)}^3", scale=ulumen.scale * 3)
Tlumen2 = Unit.create((Tlumen ** 2).dim, name="Tlumen2", dispname=f"{str(Tlumen)}^2", scale=Tlumen.scale * 2)
Tlumen3 = Unit.create((Tlumen ** 3).dim, name="Tlumen3", dispname=f"{str(Tlumen)}^3", scale=Tlumen.scale * 3)
ylumen2 = Unit.create((ylumen ** 2).dim, name="ylumen2", dispname=f"{str(ylumen)}^2", scale=ylumen.scale * 2)
ylumen3 = Unit.create((ylumen ** 3).dim, name="ylumen3", dispname=f"{str(ylumen)}^3", scale=ylumen.scale * 3)
Elumen2 = Unit.create((Elumen ** 2).dim, name="Elumen2", dispname=f"{str(Elumen)}^2", scale=Elumen.scale * 2)
Elumen3 = Unit.create((Elumen ** 3).dim, name="Elumen3", dispname=f"{str(Elumen)}^3", scale=Elumen.scale * 3)
zlumen2 = Unit.create((zlumen ** 2).dim, name="zlumen2", dispname=f"{str(zlumen)}^2", scale=zlumen.scale * 2)
zlumen3 = Unit.create((zlumen ** 3).dim, name="zlumen3", dispname=f"{str(zlumen)}^3", scale=zlumen.scale * 3)
Mlumen2 = Unit.create((Mlumen ** 2).dim, name="Mlumen2", dispname=f"{str(Mlumen)}^2", scale=Mlumen.scale * 2)
Mlumen3 = Unit.create((Mlumen ** 3).dim, name="Mlumen3", dispname=f"{str(Mlumen)}^3", scale=Mlumen.scale * 3)
klumen2 = Unit.create((klumen ** 2).dim, name="klumen2", dispname=f"{str(klumen)}^2", scale=klumen.scale * 2)
klumen3 = Unit.create((klumen ** 3).dim, name="klumen3", dispname=f"{str(klumen)}^3", scale=klumen.scale * 3)
Ylumen2 = Unit.create((Ylumen ** 2).dim, name="Ylumen2", dispname=f"{str(Ylumen)}^2", scale=Ylumen.scale * 2)
Ylumen3 = Unit.create((Ylumen ** 3).dim, name="Ylumen3", dispname=f"{str(Ylumen)}^3", scale=Ylumen.scale * 3)
alux2 = Unit.create((alux ** 2).dim, name="alux2", dispname=f"{str(alux)}^2", scale=alux.scale * 2)
alux3 = Unit.create((alux ** 3).dim, name="alux3", dispname=f"{str(alux)}^3", scale=alux.scale * 3)
clux2 = Unit.create((clux ** 2).dim, name="clux2", dispname=f"{str(clux)}^2", scale=clux.scale * 2)
clux3 = Unit.create((clux ** 3).dim, name="clux3", dispname=f"{str(clux)}^3", scale=clux.scale * 3)
Zlux2 = Unit.create((Zlux ** 2).dim, name="Zlux2", dispname=f"{str(Zlux)}^2", scale=Zlux.scale * 2)
Zlux3 = Unit.create((Zlux ** 3).dim, name="Zlux3", dispname=f"{str(Zlux)}^3", scale=Zlux.scale * 3)
Plux2 = Unit.create((Plux ** 2).dim, name="Plux2", dispname=f"{str(Plux)}^2", scale=Plux.scale * 2)
Plux3 = Unit.create((Plux ** 3).dim, name="Plux3", dispname=f"{str(Plux)}^3", scale=Plux.scale * 3)
dlux2 = Unit.create((dlux ** 2).dim, name="dlux2", dispname=f"{str(dlux)}^2", scale=dlux.scale * 2)
dlux3 = Unit.create((dlux ** 3).dim, name="dlux3", dispname=f"{str(dlux)}^3", scale=dlux.scale * 3)
Glux2 = Unit.create((Glux ** 2).dim, name="Glux2", dispname=f"{str(Glux)}^2", scale=Glux.scale * 2)
Glux3 = Unit.create((Glux ** 3).dim, name="Glux3", dispname=f"{str(Glux)}^3", scale=Glux.scale * 3)
flux2 = Unit.create((flux ** 2).dim, name="flux2", dispname=f"{str(flux)}^2", scale=flux.scale * 2)
flux3 = Unit.create((flux ** 3).dim, name="flux3", dispname=f"{str(flux)}^3", scale=flux.scale * 3)
hlux2 = Unit.create((hlux ** 2).dim, name="hlux2", dispname=f"{str(hlux)}^2", scale=hlux.scale * 2)
hlux3 = Unit.create((hlux ** 3).dim, name="hlux3", dispname=f"{str(hlux)}^3", scale=hlux.scale * 3)
dalux2 = Unit.create((dalux ** 2).dim, name="dalux2", dispname=f"{str(dalux)}^2", scale=dalux.scale * 2)
dalux3 = Unit.create((dalux ** 3).dim, name="dalux3", dispname=f"{str(dalux)}^3", scale=dalux.scale * 3)
mlux2 = Unit.create((mlux ** 2).dim, name="mlux2", dispname=f"{str(mlux)}^2", scale=mlux.scale * 2)
mlux3 = Unit.create((mlux ** 3).dim, name="mlux3", dispname=f"{str(mlux)}^3", scale=mlux.scale * 3)
nlux2 = Unit.create((nlux ** 2).dim, name="nlux2", dispname=f"{str(nlux)}^2", scale=nlux.scale * 2)
nlux3 = Unit.create((nlux ** 3).dim, name="nlux3", dispname=f"{str(nlux)}^3", scale=nlux.scale * 3)
plux2 = Unit.create((plux ** 2).dim, name="plux2", dispname=f"{str(plux)}^2", scale=plux.scale * 2)
plux3 = Unit.create((plux ** 3).dim, name="plux3", dispname=f"{str(plux)}^3", scale=plux.scale * 3)
ulux2 = Unit.create((ulux ** 2).dim, name="ulux2", dispname=f"{str(ulux)}^2", scale=ulux.scale * 2)
ulux3 = Unit.create((ulux ** 3).dim, name="ulux3", dispname=f"{str(ulux)}^3", scale=ulux.scale * 3)
Tlux2 = Unit.create((Tlux ** 2).dim, name="Tlux2", dispname=f"{str(Tlux)}^2", scale=Tlux.scale * 2)
Tlux3 = Unit.create((Tlux ** 3).dim, name="Tlux3", dispname=f"{str(Tlux)}^3", scale=Tlux.scale * 3)
ylux2 = Unit.create((ylux ** 2).dim, name="ylux2", dispname=f"{str(ylux)}^2", scale=ylux.scale * 2)
ylux3 = Unit.create((ylux ** 3).dim, name="ylux3", dispname=f"{str(ylux)}^3", scale=ylux.scale * 3)
Elux2 = Unit.create((Elux ** 2).dim, name="Elux2", dispname=f"{str(Elux)}^2", scale=Elux.scale * 2)
Elux3 = Unit.create((Elux ** 3).dim, name="Elux3", dispname=f"{str(Elux)}^3", scale=Elux.scale * 3)
zlux2 = Unit.create((zlux ** 2).dim, name="zlux2", dispname=f"{str(zlux)}^2", scale=zlux.scale * 2)
zlux3 = Unit.create((zlux ** 3).dim, name="zlux3", dispname=f"{str(zlux)}^3", scale=zlux.scale * 3)
Mlux2 = Unit.create((Mlux ** 2).dim, name="Mlux2", dispname=f"{str(Mlux)}^2", scale=Mlux.scale * 2)
Mlux3 = Unit.create((Mlux ** 3).dim, name="Mlux3", dispname=f"{str(Mlux)}^3", scale=Mlux.scale * 3)
klux2 = Unit.create((klux ** 2).dim, name="klux2", dispname=f"{str(klux)}^2", scale=klux.scale * 2)
klux3 = Unit.create((klux ** 3).dim, name="klux3", dispname=f"{str(klux)}^3", scale=klux.scale * 3)
Ylux2 = Unit.create((Ylux ** 2).dim, name="Ylux2", dispname=f"{str(Ylux)}^2", scale=Ylux.scale * 2)
Ylux3 = Unit.create((Ylux ** 3).dim, name="Ylux3", dispname=f"{str(Ylux)}^3", scale=Ylux.scale * 3)
abecquerel2 = Unit.create((abecquerel ** 2).dim, name="abecquerel2", dispname=f"{str(abecquerel)}^2",
                          scale=abecquerel.scale * 2)
abecquerel3 = Unit.create((abecquerel ** 3).dim, name="abecquerel3", dispname=f"{str(abecquerel)}^3",
                          scale=abecquerel.scale * 3)
cbecquerel2 = Unit.create((cbecquerel ** 2).dim, name="cbecquerel2", dispname=f"{str(cbecquerel)}^2",
                          scale=cbecquerel.scale * 2)
cbecquerel3 = Unit.create((cbecquerel ** 3).dim, name="cbecquerel3", dispname=f"{str(cbecquerel)}^3",
                          scale=cbecquerel.scale * 3)
Zbecquerel2 = Unit.create((Zbecquerel ** 2).dim, name="Zbecquerel2", dispname=f"{str(Zbecquerel)}^2",
                          scale=Zbecquerel.scale * 2)
Zbecquerel3 = Unit.create((Zbecquerel ** 3).dim, name="Zbecquerel3", dispname=f"{str(Zbecquerel)}^3",
                          scale=Zbecquerel.scale * 3)
Pbecquerel2 = Unit.create((Pbecquerel ** 2).dim, name="Pbecquerel2", dispname=f"{str(Pbecquerel)}^2",
                          scale=Pbecquerel.scale * 2)
Pbecquerel3 = Unit.create((Pbecquerel ** 3).dim, name="Pbecquerel3", dispname=f"{str(Pbecquerel)}^3",
                          scale=Pbecquerel.scale * 3)
dbecquerel2 = Unit.create((dbecquerel ** 2).dim, name="dbecquerel2", dispname=f"{str(dbecquerel)}^2",
                          scale=dbecquerel.scale * 2)
dbecquerel3 = Unit.create((dbecquerel ** 3).dim, name="dbecquerel3", dispname=f"{str(dbecquerel)}^3",
                          scale=dbecquerel.scale * 3)
Gbecquerel2 = Unit.create((Gbecquerel ** 2).dim, name="Gbecquerel2", dispname=f"{str(Gbecquerel)}^2",
                          scale=Gbecquerel.scale * 2)
Gbecquerel3 = Unit.create((Gbecquerel ** 3).dim, name="Gbecquerel3", dispname=f"{str(Gbecquerel)}^3",
                          scale=Gbecquerel.scale * 3)
fbecquerel2 = Unit.create((fbecquerel ** 2).dim, name="fbecquerel2", dispname=f"{str(fbecquerel)}^2",
                          scale=fbecquerel.scale * 2)
fbecquerel3 = Unit.create((fbecquerel ** 3).dim, name="fbecquerel3", dispname=f"{str(fbecquerel)}^3",
                          scale=fbecquerel.scale * 3)
hbecquerel2 = Unit.create((hbecquerel ** 2).dim, name="hbecquerel2", dispname=f"{str(hbecquerel)}^2",
                          scale=hbecquerel.scale * 2)
hbecquerel3 = Unit.create((hbecquerel ** 3).dim, name="hbecquerel3", dispname=f"{str(hbecquerel)}^3",
                          scale=hbecquerel.scale * 3)
dabecquerel2 = Unit.create((dabecquerel ** 2).dim, name="dabecquerel2", dispname=f"{str(dabecquerel)}^2",
                           scale=dabecquerel.scale * 2)
dabecquerel3 = Unit.create((dabecquerel ** 3).dim, name="dabecquerel3", dispname=f"{str(dabecquerel)}^3",
                           scale=dabecquerel.scale * 3)
mbecquerel2 = Unit.create((mbecquerel ** 2).dim, name="mbecquerel2", dispname=f"{str(mbecquerel)}^2",
                          scale=mbecquerel.scale * 2)
mbecquerel3 = Unit.create((mbecquerel ** 3).dim, name="mbecquerel3", dispname=f"{str(mbecquerel)}^3",
                          scale=mbecquerel.scale * 3)
nbecquerel2 = Unit.create((nbecquerel ** 2).dim, name="nbecquerel2", dispname=f"{str(nbecquerel)}^2",
                          scale=nbecquerel.scale * 2)
nbecquerel3 = Unit.create((nbecquerel ** 3).dim, name="nbecquerel3", dispname=f"{str(nbecquerel)}^3",
                          scale=nbecquerel.scale * 3)
pbecquerel2 = Unit.create((pbecquerel ** 2).dim, name="pbecquerel2", dispname=f"{str(pbecquerel)}^2",
                          scale=pbecquerel.scale * 2)
pbecquerel3 = Unit.create((pbecquerel ** 3).dim, name="pbecquerel3", dispname=f"{str(pbecquerel)}^3",
                          scale=pbecquerel.scale * 3)
ubecquerel2 = Unit.create((ubecquerel ** 2).dim, name="ubecquerel2", dispname=f"{str(ubecquerel)}^2",
                          scale=ubecquerel.scale * 2)
ubecquerel3 = Unit.create((ubecquerel ** 3).dim, name="ubecquerel3", dispname=f"{str(ubecquerel)}^3",
                          scale=ubecquerel.scale * 3)
Tbecquerel2 = Unit.create((Tbecquerel ** 2).dim, name="Tbecquerel2", dispname=f"{str(Tbecquerel)}^2",
                          scale=Tbecquerel.scale * 2)
Tbecquerel3 = Unit.create((Tbecquerel ** 3).dim, name="Tbecquerel3", dispname=f"{str(Tbecquerel)}^3",
                          scale=Tbecquerel.scale * 3)
ybecquerel2 = Unit.create((ybecquerel ** 2).dim, name="ybecquerel2", dispname=f"{str(ybecquerel)}^2",
                          scale=ybecquerel.scale * 2)
ybecquerel3 = Unit.create((ybecquerel ** 3).dim, name="ybecquerel3", dispname=f"{str(ybecquerel)}^3",
                          scale=ybecquerel.scale * 3)
Ebecquerel2 = Unit.create((Ebecquerel ** 2).dim, name="Ebecquerel2", dispname=f"{str(Ebecquerel)}^2",
                          scale=Ebecquerel.scale * 2)
Ebecquerel3 = Unit.create((Ebecquerel ** 3).dim, name="Ebecquerel3", dispname=f"{str(Ebecquerel)}^3",
                          scale=Ebecquerel.scale * 3)
zbecquerel2 = Unit.create((zbecquerel ** 2).dim, name="zbecquerel2", dispname=f"{str(zbecquerel)}^2",
                          scale=zbecquerel.scale * 2)
zbecquerel3 = Unit.create((zbecquerel ** 3).dim, name="zbecquerel3", dispname=f"{str(zbecquerel)}^3",
                          scale=zbecquerel.scale * 3)
Mbecquerel2 = Unit.create((Mbecquerel ** 2).dim, name="Mbecquerel2", dispname=f"{str(Mbecquerel)}^2",
                          scale=Mbecquerel.scale * 2)
Mbecquerel3 = Unit.create((Mbecquerel ** 3).dim, name="Mbecquerel3", dispname=f"{str(Mbecquerel)}^3",
                          scale=Mbecquerel.scale * 3)
kbecquerel2 = Unit.create((kbecquerel ** 2).dim, name="kbecquerel2", dispname=f"{str(kbecquerel)}^2",
                          scale=kbecquerel.scale * 2)
kbecquerel3 = Unit.create((kbecquerel ** 3).dim, name="kbecquerel3", dispname=f"{str(kbecquerel)}^3",
                          scale=kbecquerel.scale * 3)
Ybecquerel2 = Unit.create((Ybecquerel ** 2).dim, name="Ybecquerel2", dispname=f"{str(Ybecquerel)}^2",
                          scale=Ybecquerel.scale * 2)
Ybecquerel3 = Unit.create((Ybecquerel ** 3).dim, name="Ybecquerel3", dispname=f"{str(Ybecquerel)}^3",
                          scale=Ybecquerel.scale * 3)
agray2 = Unit.create((agray ** 2).dim, name="agray2", dispname=f"{str(agray)}^2", scale=agray.scale * 2)
agray3 = Unit.create((agray ** 3).dim, name="agray3", dispname=f"{str(agray)}^3", scale=agray.scale * 3)
cgray2 = Unit.create((cgray ** 2).dim, name="cgray2", dispname=f"{str(cgray)}^2", scale=cgray.scale * 2)
cgray3 = Unit.create((cgray ** 3).dim, name="cgray3", dispname=f"{str(cgray)}^3", scale=cgray.scale * 3)
Zgray2 = Unit.create((Zgray ** 2).dim, name="Zgray2", dispname=f"{str(Zgray)}^2", scale=Zgray.scale * 2)
Zgray3 = Unit.create((Zgray ** 3).dim, name="Zgray3", dispname=f"{str(Zgray)}^3", scale=Zgray.scale * 3)
Pgray2 = Unit.create((Pgray ** 2).dim, name="Pgray2", dispname=f"{str(Pgray)}^2", scale=Pgray.scale * 2)
Pgray3 = Unit.create((Pgray ** 3).dim, name="Pgray3", dispname=f"{str(Pgray)}^3", scale=Pgray.scale * 3)
dgray2 = Unit.create((dgray ** 2).dim, name="dgray2", dispname=f"{str(dgray)}^2", scale=dgray.scale * 2)
dgray3 = Unit.create((dgray ** 3).dim, name="dgray3", dispname=f"{str(dgray)}^3", scale=dgray.scale * 3)
Ggray2 = Unit.create((Ggray ** 2).dim, name="Ggray2", dispname=f"{str(Ggray)}^2", scale=Ggray.scale * 2)
Ggray3 = Unit.create((Ggray ** 3).dim, name="Ggray3", dispname=f"{str(Ggray)}^3", scale=Ggray.scale * 3)
fgray2 = Unit.create((fgray ** 2).dim, name="fgray2", dispname=f"{str(fgray)}^2", scale=fgray.scale * 2)
fgray3 = Unit.create((fgray ** 3).dim, name="fgray3", dispname=f"{str(fgray)}^3", scale=fgray.scale * 3)
hgray2 = Unit.create((hgray ** 2).dim, name="hgray2", dispname=f"{str(hgray)}^2", scale=hgray.scale * 2)
hgray3 = Unit.create((hgray ** 3).dim, name="hgray3", dispname=f"{str(hgray)}^3", scale=hgray.scale * 3)
dagray2 = Unit.create((dagray ** 2).dim, name="dagray2", dispname=f"{str(dagray)}^2", scale=dagray.scale * 2)
dagray3 = Unit.create((dagray ** 3).dim, name="dagray3", dispname=f"{str(dagray)}^3", scale=dagray.scale * 3)
mgray2 = Unit.create((mgray ** 2).dim, name="mgray2", dispname=f"{str(mgray)}^2", scale=mgray.scale * 2)
mgray3 = Unit.create((mgray ** 3).dim, name="mgray3", dispname=f"{str(mgray)}^3", scale=mgray.scale * 3)
ngray2 = Unit.create((ngray ** 2).dim, name="ngray2", dispname=f"{str(ngray)}^2", scale=ngray.scale * 2)
ngray3 = Unit.create((ngray ** 3).dim, name="ngray3", dispname=f"{str(ngray)}^3", scale=ngray.scale * 3)
pgray2 = Unit.create((pgray ** 2).dim, name="pgray2", dispname=f"{str(pgray)}^2", scale=pgray.scale * 2)
pgray3 = Unit.create((pgray ** 3).dim, name="pgray3", dispname=f"{str(pgray)}^3", scale=pgray.scale * 3)
ugray2 = Unit.create((ugray ** 2).dim, name="ugray2", dispname=f"{str(ugray)}^2", scale=ugray.scale * 2)
ugray3 = Unit.create((ugray ** 3).dim, name="ugray3", dispname=f"{str(ugray)}^3", scale=ugray.scale * 3)
Tgray2 = Unit.create((Tgray ** 2).dim, name="Tgray2", dispname=f"{str(Tgray)}^2", scale=Tgray.scale * 2)
Tgray3 = Unit.create((Tgray ** 3).dim, name="Tgray3", dispname=f"{str(Tgray)}^3", scale=Tgray.scale * 3)
ygray2 = Unit.create((ygray ** 2).dim, name="ygray2", dispname=f"{str(ygray)}^2", scale=ygray.scale * 2)
ygray3 = Unit.create((ygray ** 3).dim, name="ygray3", dispname=f"{str(ygray)}^3", scale=ygray.scale * 3)
Egray2 = Unit.create((Egray ** 2).dim, name="Egray2", dispname=f"{str(Egray)}^2", scale=Egray.scale * 2)
Egray3 = Unit.create((Egray ** 3).dim, name="Egray3", dispname=f"{str(Egray)}^3", scale=Egray.scale * 3)
zgray2 = Unit.create((zgray ** 2).dim, name="zgray2", dispname=f"{str(zgray)}^2", scale=zgray.scale * 2)
zgray3 = Unit.create((zgray ** 3).dim, name="zgray3", dispname=f"{str(zgray)}^3", scale=zgray.scale * 3)
Mgray2 = Unit.create((Mgray ** 2).dim, name="Mgray2", dispname=f"{str(Mgray)}^2", scale=Mgray.scale * 2)
Mgray3 = Unit.create((Mgray ** 3).dim, name="Mgray3", dispname=f"{str(Mgray)}^3", scale=Mgray.scale * 3)
kgray2 = Unit.create((kgray ** 2).dim, name="kgray2", dispname=f"{str(kgray)}^2", scale=kgray.scale * 2)
kgray3 = Unit.create((kgray ** 3).dim, name="kgray3", dispname=f"{str(kgray)}^3", scale=kgray.scale * 3)
Ygray2 = Unit.create((Ygray ** 2).dim, name="Ygray2", dispname=f"{str(Ygray)}^2", scale=Ygray.scale * 2)
Ygray3 = Unit.create((Ygray ** 3).dim, name="Ygray3", dispname=f"{str(Ygray)}^3", scale=Ygray.scale * 3)
asievert2 = Unit.create((asievert ** 2).dim, name="asievert2", dispname=f"{str(asievert)}^2",
                        scale=asievert.scale * 2)
asievert3 = Unit.create((asievert ** 3).dim, name="asievert3", dispname=f"{str(asievert)}^3",
                        scale=asievert.scale * 3)
csievert2 = Unit.create((csievert ** 2).dim, name="csievert2", dispname=f"{str(csievert)}^2",
                        scale=csievert.scale * 2)
csievert3 = Unit.create((csievert ** 3).dim, name="csievert3", dispname=f"{str(csievert)}^3",
                        scale=csievert.scale * 3)
Zsievert2 = Unit.create((Zsievert ** 2).dim, name="Zsievert2", dispname=f"{str(Zsievert)}^2",
                        scale=Zsievert.scale * 2)
Zsievert3 = Unit.create((Zsievert ** 3).dim, name="Zsievert3", dispname=f"{str(Zsievert)}^3",
                        scale=Zsievert.scale * 3)
Psievert2 = Unit.create((Psievert ** 2).dim, name="Psievert2", dispname=f"{str(Psievert)}^2",
                        scale=Psievert.scale * 2)
Psievert3 = Unit.create((Psievert ** 3).dim, name="Psievert3", dispname=f"{str(Psievert)}^3",
                        scale=Psievert.scale * 3)
dsievert2 = Unit.create((dsievert ** 2).dim, name="dsievert2", dispname=f"{str(dsievert)}^2",
                        scale=dsievert.scale * 2)
dsievert3 = Unit.create((dsievert ** 3).dim, name="dsievert3", dispname=f"{str(dsievert)}^3",
                        scale=dsievert.scale * 3)
Gsievert2 = Unit.create((Gsievert ** 2).dim, name="Gsievert2", dispname=f"{str(Gsievert)}^2",
                        scale=Gsievert.scale * 2)
Gsievert3 = Unit.create((Gsievert ** 3).dim, name="Gsievert3", dispname=f"{str(Gsievert)}^3",
                        scale=Gsievert.scale * 3)
fsievert2 = Unit.create((fsievert ** 2).dim, name="fsievert2", dispname=f"{str(fsievert)}^2",
                        scale=fsievert.scale * 2)
fsievert3 = Unit.create((fsievert ** 3).dim, name="fsievert3", dispname=f"{str(fsievert)}^3",
                        scale=fsievert.scale * 3)
hsievert2 = Unit.create((hsievert ** 2).dim, name="hsievert2", dispname=f"{str(hsievert)}^2",
                        scale=hsievert.scale * 2)
hsievert3 = Unit.create((hsievert ** 3).dim, name="hsievert3", dispname=f"{str(hsievert)}^3",
                        scale=hsievert.scale * 3)
dasievert2 = Unit.create((dasievert ** 2).dim, name="dasievert2", dispname=f"{str(dasievert)}^2",
                         scale=dasievert.scale * 2)
dasievert3 = Unit.create((dasievert ** 3).dim, name="dasievert3", dispname=f"{str(dasievert)}^3",
                         scale=dasievert.scale * 3)
msievert2 = Unit.create((msievert ** 2).dim, name="msievert2", dispname=f"{str(msievert)}^2",
                        scale=msievert.scale * 2)
msievert3 = Unit.create((msievert ** 3).dim, name="msievert3", dispname=f"{str(msievert)}^3",
                        scale=msievert.scale * 3)
nsievert2 = Unit.create((nsievert ** 2).dim, name="nsievert2", dispname=f"{str(nsievert)}^2",
                        scale=nsievert.scale * 2)
nsievert3 = Unit.create((nsievert ** 3).dim, name="nsievert3", dispname=f"{str(nsievert)}^3",
                        scale=nsievert.scale * 3)
psievert2 = Unit.create((psievert ** 2).dim, name="psievert2", dispname=f"{str(psievert)}^2",
                        scale=psievert.scale * 2)
psievert3 = Unit.create((psievert ** 3).dim, name="psievert3", dispname=f"{str(psievert)}^3",
                        scale=psievert.scale * 3)
usievert2 = Unit.create((usievert ** 2).dim, name="usievert2", dispname=f"{str(usievert)}^2",
                        scale=usievert.scale * 2)
usievert3 = Unit.create((usievert ** 3).dim, name="usievert3", dispname=f"{str(usievert)}^3",
                        scale=usievert.scale * 3)
Tsievert2 = Unit.create((Tsievert ** 2).dim, name="Tsievert2", dispname=f"{str(Tsievert)}^2",
                        scale=Tsievert.scale * 2)
Tsievert3 = Unit.create((Tsievert ** 3).dim, name="Tsievert3", dispname=f"{str(Tsievert)}^3",
                        scale=Tsievert.scale * 3)
ysievert2 = Unit.create((ysievert ** 2).dim, name="ysievert2", dispname=f"{str(ysievert)}^2",
                        scale=ysievert.scale * 2)
ysievert3 = Unit.create((ysievert ** 3).dim, name="ysievert3", dispname=f"{str(ysievert)}^3",
                        scale=ysievert.scale * 3)
Esievert2 = Unit.create((Esievert ** 2).dim, name="Esievert2", dispname=f"{str(Esievert)}^2",
                        scale=Esievert.scale * 2)
Esievert3 = Unit.create((Esievert ** 3).dim, name="Esievert3", dispname=f"{str(Esievert)}^3",
                        scale=Esievert.scale * 3)
zsievert2 = Unit.create((zsievert ** 2).dim, name="zsievert2", dispname=f"{str(zsievert)}^2",
                        scale=zsievert.scale * 2)
zsievert3 = Unit.create((zsievert ** 3).dim, name="zsievert3", dispname=f"{str(zsievert)}^3",
                        scale=zsievert.scale * 3)
Msievert2 = Unit.create((Msievert ** 2).dim, name="Msievert2", dispname=f"{str(Msievert)}^2",
                        scale=Msievert.scale * 2)
Msievert3 = Unit.create((Msievert ** 3).dim, name="Msievert3", dispname=f"{str(Msievert)}^3",
                        scale=Msievert.scale * 3)
ksievert2 = Unit.create((ksievert ** 2).dim, name="ksievert2", dispname=f"{str(ksievert)}^2",
                        scale=ksievert.scale * 2)
ksievert3 = Unit.create((ksievert ** 3).dim, name="ksievert3", dispname=f"{str(ksievert)}^3",
                        scale=ksievert.scale * 3)
Ysievert2 = Unit.create((Ysievert ** 2).dim, name="Ysievert2", dispname=f"{str(Ysievert)}^2",
                        scale=Ysievert.scale * 2)
Ysievert3 = Unit.create((Ysievert ** 3).dim, name="Ysievert3", dispname=f"{str(Ysievert)}^3",
                        scale=Ysievert.scale * 3)
akatal2 = Unit.create((akatal ** 2).dim, name="akatal2", dispname=f"{str(akatal)}^2", scale=akatal.scale * 2)
akatal3 = Unit.create((akatal ** 3).dim, name="akatal3", dispname=f"{str(akatal)}^3", scale=akatal.scale * 3)
ckatal2 = Unit.create((ckatal ** 2).dim, name="ckatal2", dispname=f"{str(ckatal)}^2", scale=ckatal.scale * 2)
ckatal3 = Unit.create((ckatal ** 3).dim, name="ckatal3", dispname=f"{str(ckatal)}^3", scale=ckatal.scale * 3)
Zkatal2 = Unit.create((Zkatal ** 2).dim, name="Zkatal2", dispname=f"{str(Zkatal)}^2", scale=Zkatal.scale * 2)
Zkatal3 = Unit.create((Zkatal ** 3).dim, name="Zkatal3", dispname=f"{str(Zkatal)}^3", scale=Zkatal.scale * 3)
Pkatal2 = Unit.create((Pkatal ** 2).dim, name="Pkatal2", dispname=f"{str(Pkatal)}^2", scale=Pkatal.scale * 2)
Pkatal3 = Unit.create((Pkatal ** 3).dim, name="Pkatal3", dispname=f"{str(Pkatal)}^3", scale=Pkatal.scale * 3)
dkatal2 = Unit.create((dkatal ** 2).dim, name="dkatal2", dispname=f"{str(dkatal)}^2", scale=dkatal.scale * 2)
dkatal3 = Unit.create((dkatal ** 3).dim, name="dkatal3", dispname=f"{str(dkatal)}^3", scale=dkatal.scale * 3)
Gkatal2 = Unit.create((Gkatal ** 2).dim, name="Gkatal2", dispname=f"{str(Gkatal)}^2", scale=Gkatal.scale * 2)
Gkatal3 = Unit.create((Gkatal ** 3).dim, name="Gkatal3", dispname=f"{str(Gkatal)}^3", scale=Gkatal.scale * 3)
fkatal2 = Unit.create((fkatal ** 2).dim, name="fkatal2", dispname=f"{str(fkatal)}^2", scale=fkatal.scale * 2)
fkatal3 = Unit.create((fkatal ** 3).dim, name="fkatal3", dispname=f"{str(fkatal)}^3", scale=fkatal.scale * 3)
hkatal2 = Unit.create((hkatal ** 2).dim, name="hkatal2", dispname=f"{str(hkatal)}^2", scale=hkatal.scale * 2)
hkatal3 = Unit.create((hkatal ** 3).dim, name="hkatal3", dispname=f"{str(hkatal)}^3", scale=hkatal.scale * 3)
dakatal2 = Unit.create((dakatal ** 2).dim, name="dakatal2", dispname=f"{str(dakatal)}^2", scale=dakatal.scale * 2)
dakatal3 = Unit.create((dakatal ** 3).dim, name="dakatal3", dispname=f"{str(dakatal)}^3", scale=dakatal.scale * 3)
mkatal2 = Unit.create((mkatal ** 2).dim, name="mkatal2", dispname=f"{str(mkatal)}^2", scale=mkatal.scale * 2)
mkatal3 = Unit.create((mkatal ** 3).dim, name="mkatal3", dispname=f"{str(mkatal)}^3", scale=mkatal.scale * 3)
nkatal2 = Unit.create((nkatal ** 2).dim, name="nkatal2", dispname=f"{str(nkatal)}^2", scale=nkatal.scale * 2)
nkatal3 = Unit.create((nkatal ** 3).dim, name="nkatal3", dispname=f"{str(nkatal)}^3", scale=nkatal.scale * 3)
pkatal2 = Unit.create((pkatal ** 2).dim, name="pkatal2", dispname=f"{str(pkatal)}^2", scale=pkatal.scale * 2)
pkatal3 = Unit.create((pkatal ** 3).dim, name="pkatal3", dispname=f"{str(pkatal)}^3", scale=pkatal.scale * 3)
ukatal2 = Unit.create((ukatal ** 2).dim, name="ukatal2", dispname=f"{str(ukatal)}^2", scale=ukatal.scale * 2)
ukatal3 = Unit.create((ukatal ** 3).dim, name="ukatal3", dispname=f"{str(ukatal)}^3", scale=ukatal.scale * 3)
Tkatal2 = Unit.create((Tkatal ** 2).dim, name="Tkatal2", dispname=f"{str(Tkatal)}^2", scale=Tkatal.scale * 2)
Tkatal3 = Unit.create((Tkatal ** 3).dim, name="Tkatal3", dispname=f"{str(Tkatal)}^3", scale=Tkatal.scale * 3)
ykatal2 = Unit.create((ykatal ** 2).dim, name="ykatal2", dispname=f"{str(ykatal)}^2", scale=ykatal.scale * 2)
ykatal3 = Unit.create((ykatal ** 3).dim, name="ykatal3", dispname=f"{str(ykatal)}^3", scale=ykatal.scale * 3)
Ekatal2 = Unit.create((Ekatal ** 2).dim, name="Ekatal2", dispname=f"{str(Ekatal)}^2", scale=Ekatal.scale * 2)
Ekatal3 = Unit.create((Ekatal ** 3).dim, name="Ekatal3", dispname=f"{str(Ekatal)}^3", scale=Ekatal.scale * 3)
zkatal2 = Unit.create((zkatal ** 2).dim, name="zkatal2", dispname=f"{str(zkatal)}^2", scale=zkatal.scale * 2)
zkatal3 = Unit.create((zkatal ** 3).dim, name="zkatal3", dispname=f"{str(zkatal)}^3", scale=zkatal.scale * 3)
Mkatal2 = Unit.create((Mkatal ** 2).dim, name="Mkatal2", dispname=f"{str(Mkatal)}^2", scale=Mkatal.scale * 2)
Mkatal3 = Unit.create((Mkatal ** 3).dim, name="Mkatal3", dispname=f"{str(Mkatal)}^3", scale=Mkatal.scale * 3)
kkatal2 = Unit.create((kkatal ** 2).dim, name="kkatal2", dispname=f"{str(kkatal)}^2", scale=kkatal.scale * 2)
kkatal3 = Unit.create((kkatal ** 3).dim, name="kkatal3", dispname=f"{str(kkatal)}^3", scale=kkatal.scale * 3)
Ykatal2 = Unit.create((Ykatal ** 2).dim, name="Ykatal2", dispname=f"{str(Ykatal)}^2", scale=Ykatal.scale * 2)
Ykatal3 = Unit.create((Ykatal ** 3).dim, name="Ykatal3", dispname=f"{str(Ykatal)}^3", scale=Ykatal.scale * 3)
aliter = Unit.create_scaled_unit(liter, "a")
liter = Unit.create_scaled_unit(liter, "")
cliter = Unit.create_scaled_unit(liter, "c")
Zliter = Unit.create_scaled_unit(liter, "Z")
Pliter = Unit.create_scaled_unit(liter, "P")
dliter = Unit.create_scaled_unit(liter, "d")
Gliter = Unit.create_scaled_unit(liter, "G")
fliter = Unit.create_scaled_unit(liter, "f")
hliter = Unit.create_scaled_unit(liter, "h")
daliter = Unit.create_scaled_unit(liter, "da")
mliter = Unit.create_scaled_unit(liter, "m")
nliter = Unit.create_scaled_unit(liter, "n")
pliter = Unit.create_scaled_unit(liter, "p")
uliter = Unit.create_scaled_unit(liter, "u")
Tliter = Unit.create_scaled_unit(liter, "T")
yliter = Unit.create_scaled_unit(liter, "y")
Eliter = Unit.create_scaled_unit(liter, "E")
zliter = Unit.create_scaled_unit(liter, "z")
Mliter = Unit.create_scaled_unit(liter, "M")
kliter = Unit.create_scaled_unit(liter, "k")
Yliter = Unit.create_scaled_unit(liter, "Y")
alitre = Unit.create_scaled_unit(litre, "a")
litre = Unit.create_scaled_unit(litre, "")
clitre = Unit.create_scaled_unit(litre, "c")
Zlitre = Unit.create_scaled_unit(litre, "Z")
Plitre = Unit.create_scaled_unit(litre, "P")
dlitre = Unit.create_scaled_unit(litre, "d")
Glitre = Unit.create_scaled_unit(litre, "G")
flitre = Unit.create_scaled_unit(litre, "f")
hlitre = Unit.create_scaled_unit(litre, "h")
dalitre = Unit.create_scaled_unit(litre, "da")
mlitre = Unit.create_scaled_unit(litre, "m")
nlitre = Unit.create_scaled_unit(litre, "n")
plitre = Unit.create_scaled_unit(litre, "p")
ulitre = Unit.create_scaled_unit(litre, "u")
Tlitre = Unit.create_scaled_unit(litre, "T")
ylitre = Unit.create_scaled_unit(litre, "y")
Elitre = Unit.create_scaled_unit(litre, "E")
zlitre = Unit.create_scaled_unit(litre, "z")
Mlitre = Unit.create_scaled_unit(litre, "M")
klitre = Unit.create_scaled_unit(litre, "k")
Ylitre = Unit.create_scaled_unit(litre, "Y")

base_units = [
  katal,
  sievert,
  gray,
  becquerel,
  lux,
  lumen,
  henry,
  tesla,
  weber,
  siemens,
  ohm,
  farad,
  volt,
  coulomb,
  watt,
  joule,
  pascal,
  newton,
  hertz,
  steradian,
  radian,
  molar,
  gramme,
  gram,
  kilogramme,
  candle,
  mol,
  mole,
  kelvin,
  ampere,
  amp,
  second,
  kilogram,
  meter,
  metre,
]

scaled_units = [
  Ykatal,
  kkatal,
  Mkatal,
  zkatal,
  Ekatal,
  ykatal,
  Tkatal,
  ukatal,
  pkatal,
  nkatal,
  mkatal,
  fkatal,
  Gkatal,
  Pkatal,
  Zkatal,
  akatal,
  Ysievert,
  ksievert,
  Msievert,
  zsievert,
  Esievert,
  ysievert,
  Tsievert,
  usievert,
  psievert,
  nsievert,
  msievert,
  fsievert,
  Gsievert,
  Psievert,
  Zsievert,
  asievert,
  Ygray,
  kgray,
  Mgray,
  zgray,
  Egray,
  ygray,
  Tgray,
  ugray,
  pgray,
  ngray,
  mgray,
  fgray,
  Ggray,
  Pgray,
  Zgray,
  agray,
  Ybecquerel,
  kbecquerel,
  Mbecquerel,
  zbecquerel,
  Ebecquerel,
  ybecquerel,
  Tbecquerel,
  ubecquerel,
  pbecquerel,
  nbecquerel,
  mbecquerel,
  fbecquerel,
  Gbecquerel,
  Pbecquerel,
  Zbecquerel,
  abecquerel,
  Ylux,
  klux,
  Mlux,
  zlux,
  Elux,
  ylux,
  Tlux,
  ulux,
  plux,
  nlux,
  mlux,
  flux,
  Glux,
  Plux,
  Zlux,
  alux,
  Ylumen,
  klumen,
  Mlumen,
  zlumen,
  Elumen,
  ylumen,
  Tlumen,
  ulumen,
  plumen,
  nlumen,
  mlumen,
  flumen,
  Glumen,
  Plumen,
  Zlumen,
  alumen,
  Yhenry,
  khenry,
  Mhenry,
  zhenry,
  Ehenry,
  yhenry,
  Thenry,
  uhenry,
  phenry,
  nhenry,
  mhenry,
  fhenry,
  Ghenry,
  Phenry,
  Zhenry,
  ahenry,
  Ytesla,
  ktesla,
  Mtesla,
  ztesla,
  Etesla,
  ytesla,
  Ttesla,
  utesla,
  ptesla,
  ntesla,
  mtesla,
  ftesla,
  Gtesla,
  Ptesla,
  Ztesla,
  atesla,
  Yweber,
  kweber,
  Mweber,
  zweber,
  Eweber,
  yweber,
  Tweber,
  uweber,
  pweber,
  nweber,
  mweber,
  fweber,
  Gweber,
  Pweber,
  Zweber,
  aweber,
  Ysiemens,
  ksiemens,
  Msiemens,
  zsiemens,
  Esiemens,
  ysiemens,
  Tsiemens,
  usiemens,
  psiemens,
  nsiemens,
  msiemens,
  fsiemens,
  Gsiemens,
  Psiemens,
  Zsiemens,
  asiemens,
  Yohm,
  kohm,
  Mohm,
  zohm,
  Eohm,
  yohm,
  Tohm,
  uohm,
  pohm,
  nohm,
  mohm,
  fohm,
  Gohm,
  Pohm,
  Zohm,
  aohm,
  Yfarad,
  kfarad,
  Mfarad,
  zfarad,
  Efarad,
  yfarad,
  Tfarad,
  ufarad,
  pfarad,
  nfarad,
  mfarad,
  ffarad,
  Gfarad,
  Pfarad,
  Zfarad,
  afarad,
  Yvolt,
  kvolt,
  Mvolt,
  zvolt,
  Evolt,
  yvolt,
  Tvolt,
  uvolt,
  pvolt,
  nvolt,
  mvolt,
  fvolt,
  Gvolt,
  Pvolt,
  Zvolt,
  avolt,
  Ycoulomb,
  kcoulomb,
  Mcoulomb,
  zcoulomb,
  Ecoulomb,
  ycoulomb,
  Tcoulomb,
  ucoulomb,
  pcoulomb,
  ncoulomb,
  mcoulomb,
  fcoulomb,
  Gcoulomb,
  Pcoulomb,
  Zcoulomb,
  acoulomb,
  Ywatt,
  kwatt,
  Mwatt,
  zwatt,
  Ewatt,
  ywatt,
  Twatt,
  uwatt,
  pwatt,
  nwatt,
  mwatt,
  fwatt,
  Gwatt,
  Pwatt,
  Zwatt,
  awatt,
  Yjoule,
  kjoule,
  Mjoule,
  zjoule,
  Ejoule,
  yjoule,
  Tjoule,
  ujoule,
  pjoule,
  njoule,
  mjoule,
  fjoule,
  Gjoule,
  Pjoule,
  Zjoule,
  ajoule,
  Ypascal,
  kpascal,
  Mpascal,
  zpascal,
  Epascal,
  ypascal,
  Tpascal,
  upascal,
  ppascal,
  npascal,
  mpascal,
  fpascal,
  Gpascal,
  Ppascal,
  Zpascal,
  apascal,
  Ynewton,
  knewton,
  Mnewton,
  znewton,
  Enewton,
  ynewton,
  Tnewton,
  unewton,
  pnewton,
  nnewton,
  mnewton,
  fnewton,
  Gnewton,
  Pnewton,
  Znewton,
  anewton,
  Yhertz,
  khertz,
  Mhertz,
  zhertz,
  Ehertz,
  yhertz,
  Thertz,
  uhertz,
  phertz,
  nhertz,
  mhertz,
  fhertz,
  Ghertz,
  Phertz,
  Zhertz,
  ahertz,
  Ysteradian,
  ksteradian,
  Msteradian,
  zsteradian,
  Esteradian,
  ysteradian,
  Tsteradian,
  usteradian,
  psteradian,
  nsteradian,
  msteradian,
  fsteradian,
  Gsteradian,
  Psteradian,
  Zsteradian,
  asteradian,
  Yradian,
  kradian,
  Mradian,
  zradian,
  Eradian,
  yradian,
  Tradian,
  uradian,
  pradian,
  nradian,
  mradian,
  fradian,
  Gradian,
  Pradian,
  Zradian,
  aradian,
  Ymolar,
  kmolar,
  Mmolar,
  zmolar,
  Emolar,
  ymolar,
  Tmolar,
  umolar,
  pmolar,
  nmolar,
  mmolar,
  fmolar,
  Gmolar,
  Pmolar,
  Zmolar,
  amolar,
  Ygramme,
  kgramme,
  Mgramme,
  zgramme,
  Egramme,
  ygramme,
  Tgramme,
  ugramme,
  pgramme,
  ngramme,
  mgramme,
  fgramme,
  Ggramme,
  Pgramme,
  Zgramme,
  agramme,
  Ygram,
  kgram,
  Mgram,
  zgram,
  Egram,
  ygram,
  Tgram,
  ugram,
  pgram,
  ngram,
  mgram,
  fgram,
  Ggram,
  Pgram,
  Zgram,
  agram,
  Ycandle,
  kcandle,
  Mcandle,
  zcandle,
  Ecandle,
  ycandle,
  Tcandle,
  ucandle,
  pcandle,
  ncandle,
  mcandle,
  fcandle,
  Gcandle,
  Pcandle,
  Zcandle,
  acandle,
  Ymol,
  kmol,
  Mmol,
  zmol,
  Emol,
  ymol,
  Tmol,
  umol,
  pmol,
  nmol,
  mmol,
  fmol,
  Gmol,
  Pmol,
  Zmol,
  amol,
  Ymole,
  kmole,
  Mmole,
  zmole,
  Emole,
  ymole,
  Tmole,
  umole,
  pmole,
  nmole,
  mmole,
  fmole,
  Gmole,
  Pmole,
  Zmole,
  amole,
  Yampere,
  kampere,
  Mampere,
  zampere,
  Eampere,
  yampere,
  Tampere,
  uampere,
  pampere,
  nampere,
  mampere,
  fampere,
  Gampere,
  Pampere,
  Zampere,
  aampere,
  Yamp,
  kamp,
  Mamp,
  zamp,
  Eamp,
  yamp,
  Tamp,
  uamp,
  pamp,
  namp,
  mamp,
  famp,
  Gamp,
  Pamp,
  Zamp,
  aamp,
  Ysecond,
  ksecond,
  Msecond,
  zsecond,
  Esecond,
  ysecond,
  Tsecond,
  usecond,
  psecond,
  nsecond,
  msecond,
  fsecond,
  Gsecond,
  Psecond,
  Zsecond,
  asecond,
  Ymeter,
  kmeter,
  Mmeter,
  zmeter,
  Emeter,
  ymeter,
  Tmeter,
  umeter,
  pmeter,
  nmeter,
  mmeter,
  fmeter,
  Gmeter,
  Pmeter,
  Zmeter,
  ameter,
  Ymetre,
  kmetre,
  Mmetre,
  zmetre,
  Emetre,
  ymetre,
  Tmetre,
  umetre,
  pmetre,
  nmetre,
  mmetre,
  fmetre,
  Gmetre,
  Pmetre,
  Zmetre,
  ametre,
]

powered_units = [
  Ykatal3,
  Ykatal2,
  kkatal3,
  kkatal2,
  Mkatal3,
  Mkatal2,
  zkatal3,
  zkatal2,
  Ekatal3,
  Ekatal2,
  ykatal3,
  ykatal2,
  Tkatal3,
  Tkatal2,
  ukatal3,
  ukatal2,
  pkatal3,
  pkatal2,
  nkatal3,
  nkatal2,
  mkatal3,
  mkatal2,
  fkatal3,
  fkatal2,
  Gkatal3,
  Gkatal2,
  Pkatal3,
  Pkatal2,
  Zkatal3,
  Zkatal2,
  akatal3,
  akatal2,
  Ysievert3,
  Ysievert2,
  ksievert3,
  ksievert2,
  Msievert3,
  Msievert2,
  zsievert3,
  zsievert2,
  Esievert3,
  Esievert2,
  ysievert3,
  ysievert2,
  Tsievert3,
  Tsievert2,
  usievert3,
  usievert2,
  psievert3,
  psievert2,
  nsievert3,
  nsievert2,
  msievert3,
  msievert2,
  fsievert3,
  fsievert2,
  Gsievert3,
  Gsievert2,
  Psievert3,
  Psievert2,
  Zsievert3,
  Zsievert2,
  asievert3,
  asievert2,
  Ygray3,
  Ygray2,
  kgray3,
  kgray2,
  Mgray3,
  Mgray2,
  zgray3,
  zgray2,
  Egray3,
  Egray2,
  ygray3,
  ygray2,
  Tgray3,
  Tgray2,
  ugray3,
  ugray2,
  pgray3,
  pgray2,
  ngray3,
  ngray2,
  mgray3,
  mgray2,
  fgray3,
  fgray2,
  Ggray3,
  Ggray2,
  Pgray3,
  Pgray2,
  Zgray3,
  Zgray2,
  agray3,
  agray2,
  Ybecquerel3,
  Ybecquerel2,
  kbecquerel3,
  kbecquerel2,
  Mbecquerel3,
  Mbecquerel2,
  zbecquerel3,
  zbecquerel2,
  Ebecquerel3,
  Ebecquerel2,
  ybecquerel3,
  ybecquerel2,
  Tbecquerel3,
  Tbecquerel2,
  ubecquerel3,
  ubecquerel2,
  pbecquerel3,
  pbecquerel2,
  nbecquerel3,
  nbecquerel2,
  mbecquerel3,
  mbecquerel2,
  fbecquerel3,
  fbecquerel2,
  Gbecquerel3,
  Gbecquerel2,
  Pbecquerel3,
  Pbecquerel2,
  Zbecquerel3,
  Zbecquerel2,
  abecquerel3,
  abecquerel2,
  Ylux3,
  Ylux2,
  klux3,
  klux2,
  Mlux3,
  Mlux2,
  zlux3,
  zlux2,
  Elux3,
  Elux2,
  ylux3,
  ylux2,
  Tlux3,
  Tlux2,
  ulux3,
  ulux2,
  plux3,
  plux2,
  nlux3,
  nlux2,
  mlux3,
  mlux2,
  flux3,
  flux2,
  Glux3,
  Glux2,
  Plux3,
  Plux2,
  Zlux3,
  Zlux2,
  alux3,
  alux2,
  Ylumen3,
  Ylumen2,
  klumen3,
  klumen2,
  Mlumen3,
  Mlumen2,
  zlumen3,
  zlumen2,
  Elumen3,
  Elumen2,
  ylumen3,
  ylumen2,
  Tlumen3,
  Tlumen2,
  ulumen3,
  ulumen2,
  plumen3,
  plumen2,
  nlumen3,
  nlumen2,
  mlumen3,
  mlumen2,
  flumen3,
  flumen2,
  Glumen3,
  Glumen2,
  Plumen3,
  Plumen2,
  Zlumen3,
  Zlumen2,
  alumen3,
  alumen2,
  Yhenry3,
  Yhenry2,
  khenry3,
  khenry2,
  Mhenry3,
  Mhenry2,
  zhenry3,
  zhenry2,
  Ehenry3,
  Ehenry2,
  yhenry3,
  yhenry2,
  Thenry3,
  Thenry2,
  uhenry3,
  uhenry2,
  phenry3,
  phenry2,
  nhenry3,
  nhenry2,
  mhenry3,
  mhenry2,
  fhenry3,
  fhenry2,
  Ghenry3,
  Ghenry2,
  Phenry3,
  Phenry2,
  Zhenry3,
  Zhenry2,
  ahenry3,
  ahenry2,
  Ytesla3,
  Ytesla2,
  ktesla3,
  ktesla2,
  Mtesla3,
  Mtesla2,
  ztesla3,
  ztesla2,
  Etesla3,
  Etesla2,
  ytesla3,
  ytesla2,
  Ttesla3,
  Ttesla2,
  utesla3,
  utesla2,
  ptesla3,
  ptesla2,
  ntesla3,
  ntesla2,
  mtesla3,
  mtesla2,
  ftesla3,
  ftesla2,
  Gtesla3,
  Gtesla2,
  Ptesla3,
  Ptesla2,
  Ztesla3,
  Ztesla2,
  atesla3,
  atesla2,
  Yweber3,
  Yweber2,
  kweber3,
  kweber2,
  Mweber3,
  Mweber2,
  zweber3,
  zweber2,
  Eweber3,
  Eweber2,
  yweber3,
  yweber2,
  Tweber3,
  Tweber2,
  uweber3,
  uweber2,
  pweber3,
  pweber2,
  nweber3,
  nweber2,
  mweber3,
  mweber2,
  fweber3,
  fweber2,
  Gweber3,
  Gweber2,
  Pweber3,
  Pweber2,
  Zweber3,
  Zweber2,
  aweber3,
  aweber2,
  Ysiemens3,
  Ysiemens2,
  ksiemens3,
  ksiemens2,
  Msiemens3,
  Msiemens2,
  zsiemens3,
  zsiemens2,
  Esiemens3,
  Esiemens2,
  ysiemens3,
  ysiemens2,
  Tsiemens3,
  Tsiemens2,
  usiemens3,
  usiemens2,
  psiemens3,
  psiemens2,
  nsiemens3,
  nsiemens2,
  msiemens3,
  msiemens2,
  fsiemens3,
  fsiemens2,
  Gsiemens3,
  Gsiemens2,
  Psiemens3,
  Psiemens2,
  Zsiemens3,
  Zsiemens2,
  asiemens3,
  asiemens2,
  Yohm3,
  Yohm2,
  kohm3,
  kohm2,
  Mohm3,
  Mohm2,
  zohm3,
  zohm2,
  Eohm3,
  Eohm2,
  yohm3,
  yohm2,
  Tohm3,
  Tohm2,
  uohm3,
  uohm2,
  pohm3,
  pohm2,
  nohm3,
  nohm2,
  mohm3,
  mohm2,
  fohm3,
  fohm2,
  Gohm3,
  Gohm2,
  Pohm3,
  Pohm2,
  Zohm3,
  Zohm2,
  aohm3,
  aohm2,
  Yfarad3,
  Yfarad2,
  kfarad3,
  kfarad2,
  Mfarad3,
  Mfarad2,
  zfarad3,
  zfarad2,
  Efarad3,
  Efarad2,
  yfarad3,
  yfarad2,
  Tfarad3,
  Tfarad2,
  ufarad3,
  ufarad2,
  pfarad3,
  pfarad2,
  nfarad3,
  nfarad2,
  mfarad3,
  mfarad2,
  ffarad3,
  ffarad2,
  Gfarad3,
  Gfarad2,
  Pfarad3,
  Pfarad2,
  Zfarad3,
  Zfarad2,
  afarad3,
  afarad2,
  Yvolt3,
  Yvolt2,
  kvolt3,
  kvolt2,
  Mvolt3,
  Mvolt2,
  zvolt3,
  zvolt2,
  Evolt3,
  Evolt2,
  yvolt3,
  yvolt2,
  Tvolt3,
  Tvolt2,
  uvolt3,
  uvolt2,
  pvolt3,
  pvolt2,
  nvolt3,
  nvolt2,
  mvolt3,
  mvolt2,
  fvolt3,
  fvolt2,
  Gvolt3,
  Gvolt2,
  Pvolt3,
  Pvolt2,
  Zvolt3,
  Zvolt2,
  avolt3,
  avolt2,
  Ycoulomb3,
  Ycoulomb2,
  kcoulomb3,
  kcoulomb2,
  Mcoulomb3,
  Mcoulomb2,
  zcoulomb3,
  zcoulomb2,
  Ecoulomb3,
  Ecoulomb2,
  ycoulomb3,
  ycoulomb2,
  Tcoulomb3,
  Tcoulomb2,
  ucoulomb3,
  ucoulomb2,
  pcoulomb3,
  pcoulomb2,
  ncoulomb3,
  ncoulomb2,
  mcoulomb3,
  mcoulomb2,
  fcoulomb3,
  fcoulomb2,
  Gcoulomb3,
  Gcoulomb2,
  Pcoulomb3,
  Pcoulomb2,
  Zcoulomb3,
  Zcoulomb2,
  acoulomb3,
  acoulomb2,
  Ywatt3,
  Ywatt2,
  kwatt3,
  kwatt2,
  Mwatt3,
  Mwatt2,
  zwatt3,
  zwatt2,
  Ewatt3,
  Ewatt2,
  ywatt3,
  ywatt2,
  Twatt3,
  Twatt2,
  uwatt3,
  uwatt2,
  pwatt3,
  pwatt2,
  nwatt3,
  nwatt2,
  mwatt3,
  mwatt2,
  fwatt3,
  fwatt2,
  Gwatt3,
  Gwatt2,
  Pwatt3,
  Pwatt2,
  Zwatt3,
  Zwatt2,
  awatt3,
  awatt2,
  Yjoule3,
  Yjoule2,
  kjoule3,
  kjoule2,
  Mjoule3,
  Mjoule2,
  zjoule3,
  zjoule2,
  Ejoule3,
  Ejoule2,
  yjoule3,
  yjoule2,
  Tjoule3,
  Tjoule2,
  ujoule3,
  ujoule2,
  pjoule3,
  pjoule2,
  njoule3,
  njoule2,
  mjoule3,
  mjoule2,
  fjoule3,
  fjoule2,
  Gjoule3,
  Gjoule2,
  Pjoule3,
  Pjoule2,
  Zjoule3,
  Zjoule2,
  ajoule3,
  ajoule2,
  Ypascal3,
  Ypascal2,
  kpascal3,
  kpascal2,
  Mpascal3,
  Mpascal2,
  zpascal3,
  zpascal2,
  Epascal3,
  Epascal2,
  ypascal3,
  ypascal2,
  Tpascal3,
  Tpascal2,
  upascal3,
  upascal2,
  ppascal3,
  ppascal2,
  npascal3,
  npascal2,
  mpascal3,
  mpascal2,
  fpascal3,
  fpascal2,
  Gpascal3,
  Gpascal2,
  Ppascal3,
  Ppascal2,
  Zpascal3,
  Zpascal2,
  apascal3,
  apascal2,
  Ynewton3,
  Ynewton2,
  knewton3,
  knewton2,
  Mnewton3,
  Mnewton2,
  znewton3,
  znewton2,
  Enewton3,
  Enewton2,
  ynewton3,
  ynewton2,
  Tnewton3,
  Tnewton2,
  unewton3,
  unewton2,
  pnewton3,
  pnewton2,
  nnewton3,
  nnewton2,
  mnewton3,
  mnewton2,
  fnewton3,
  fnewton2,
  Gnewton3,
  Gnewton2,
  Pnewton3,
  Pnewton2,
  Znewton3,
  Znewton2,
  anewton3,
  anewton2,
  Yhertz3,
  Yhertz2,
  khertz3,
  khertz2,
  Mhertz3,
  Mhertz2,
  zhertz3,
  zhertz2,
  Ehertz3,
  Ehertz2,
  yhertz3,
  yhertz2,
  Thertz3,
  Thertz2,
  uhertz3,
  uhertz2,
  phertz3,
  phertz2,
  nhertz3,
  nhertz2,
  mhertz3,
  mhertz2,
  fhertz3,
  fhertz2,
  Ghertz3,
  Ghertz2,
  Phertz3,
  Phertz2,
  Zhertz3,
  Zhertz2,
  ahertz3,
  ahertz2,
  Ysteradian3,
  Ysteradian2,
  ksteradian3,
  ksteradian2,
  Msteradian3,
  Msteradian2,
  zsteradian3,
  zsteradian2,
  Esteradian3,
  Esteradian2,
  ysteradian3,
  ysteradian2,
  Tsteradian3,
  Tsteradian2,
  usteradian3,
  usteradian2,
  psteradian3,
  psteradian2,
  nsteradian3,
  nsteradian2,
  msteradian3,
  msteradian2,
  fsteradian3,
  fsteradian2,
  Gsteradian3,
  Gsteradian2,
  Psteradian3,
  Psteradian2,
  Zsteradian3,
  Zsteradian2,
  asteradian3,
  asteradian2,
  Yradian3,
  Yradian2,
  kradian3,
  kradian2,
  Mradian3,
  Mradian2,
  zradian3,
  zradian2,
  Eradian3,
  Eradian2,
  yradian3,
  yradian2,
  Tradian3,
  Tradian2,
  uradian3,
  uradian2,
  pradian3,
  pradian2,
  nradian3,
  nradian2,
  mradian3,
  mradian2,
  fradian3,
  fradian2,
  Gradian3,
  Gradian2,
  Pradian3,
  Pradian2,
  Zradian3,
  Zradian2,
  aradian3,
  aradian2,
  Ymolar3,
  Ymolar2,
  kmolar3,
  kmolar2,
  Mmolar3,
  Mmolar2,
  zmolar3,
  zmolar2,
  Emolar3,
  Emolar2,
  ymolar3,
  ymolar2,
  Tmolar3,
  Tmolar2,
  umolar3,
  umolar2,
  pmolar3,
  pmolar2,
  nmolar3,
  nmolar2,
  mmolar3,
  mmolar2,
  fmolar3,
  fmolar2,
  Gmolar3,
  Gmolar2,
  Pmolar3,
  Pmolar2,
  Zmolar3,
  Zmolar2,
  amolar3,
  amolar2,
  Ygramme3,
  Ygramme2,
  kgramme3,
  kgramme2,
  Mgramme3,
  Mgramme2,
  zgramme3,
  zgramme2,
  Egramme3,
  Egramme2,
  ygramme3,
  ygramme2,
  Tgramme3,
  Tgramme2,
  ugramme3,
  ugramme2,
  pgramme3,
  pgramme2,
  ngramme3,
  ngramme2,
  mgramme3,
  mgramme2,
  fgramme3,
  fgramme2,
  Ggramme3,
  Ggramme2,
  Pgramme3,
  Pgramme2,
  Zgramme3,
  Zgramme2,
  agramme3,
  agramme2,
  Ygram3,
  Ygram2,
  kgram3,
  kgram2,
  Mgram3,
  Mgram2,
  zgram3,
  zgram2,
  Egram3,
  Egram2,
  ygram3,
  ygram2,
  Tgram3,
  Tgram2,
  ugram3,
  ugram2,
  pgram3,
  pgram2,
  ngram3,
  ngram2,
  mgram3,
  mgram2,
  fgram3,
  fgram2,
  Ggram3,
  Ggram2,
  Pgram3,
  Pgram2,
  Zgram3,
  Zgram2,
  agram3,
  agram2,
  Ycandle3,
  Ycandle2,
  kcandle3,
  kcandle2,
  Mcandle3,
  Mcandle2,
  zcandle3,
  zcandle2,
  Ecandle3,
  Ecandle2,
  ycandle3,
  ycandle2,
  Tcandle3,
  Tcandle2,
  ucandle3,
  ucandle2,
  pcandle3,
  pcandle2,
  ncandle3,
  ncandle2,
  mcandle3,
  mcandle2,
  fcandle3,
  fcandle2,
  Gcandle3,
  Gcandle2,
  Pcandle3,
  Pcandle2,
  Zcandle3,
  Zcandle2,
  acandle3,
  acandle2,
  Ymol3,
  Ymol2,
  kmol3,
  kmol2,
  Mmol3,
  Mmol2,
  zmol3,
  zmol2,
  Emol3,
  Emol2,
  ymol3,
  ymol2,
  Tmol3,
  Tmol2,
  umol3,
  umol2,
  pmol3,
  pmol2,
  nmol3,
  nmol2,
  mmol3,
  mmol2,
  fmol3,
  fmol2,
  Gmol3,
  Gmol2,
  Pmol3,
  Pmol2,
  Zmol3,
  Zmol2,
  amol3,
  amol2,
  Ymole3,
  Ymole2,
  kmole3,
  kmole2,
  Mmole3,
  Mmole2,
  zmole3,
  zmole2,
  Emole3,
  Emole2,
  ymole3,
  ymole2,
  Tmole3,
  Tmole2,
  umole3,
  umole2,
  pmole3,
  pmole2,
  nmole3,
  nmole2,
  mmole3,
  mmole2,
  fmole3,
  fmole2,
  Gmole3,
  Gmole2,
  Pmole3,
  Pmole2,
  Zmole3,
  Zmole2,
  amole3,
  amole2,
  Yampere3,
  Yampere2,
  kampere3,
  kampere2,
  Mampere3,
  Mampere2,
  zampere3,
  zampere2,
  Eampere3,
  Eampere2,
  yampere3,
  yampere2,
  Tampere3,
  Tampere2,
  uampere3,
  uampere2,
  pampere3,
  pampere2,
  nampere3,
  nampere2,
  mampere3,
  mampere2,
  fampere3,
  fampere2,
  Gampere3,
  Gampere2,
  Pampere3,
  Pampere2,
  Zampere3,
  Zampere2,
  aampere3,
  aampere2,
  Yamp3,
  Yamp2,
  kamp3,
  kamp2,
  Mamp3,
  Mamp2,
  zamp3,
  zamp2,
  Eamp3,
  Eamp2,
  yamp3,
  yamp2,
  Tamp3,
  Tamp2,
  uamp3,
  uamp2,
  pamp3,
  pamp2,
  namp3,
  namp2,
  mamp3,
  mamp2,
  famp3,
  famp2,
  Gamp3,
  Gamp2,
  Pamp3,
  Pamp2,
  Zamp3,
  Zamp2,
  aamp3,
  aamp2,
  Ysecond3,
  Ysecond2,
  ksecond3,
  ksecond2,
  Msecond3,
  Msecond2,
  zsecond3,
  zsecond2,
  Esecond3,
  Esecond2,
  ysecond3,
  ysecond2,
  Tsecond3,
  Tsecond2,
  usecond3,
  usecond2,
  psecond3,
  psecond2,
  nsecond3,
  nsecond2,
  msecond3,
  msecond2,
  fsecond3,
  fsecond2,
  Gsecond3,
  Gsecond2,
  Psecond3,
  Psecond2,
  Zsecond3,
  Zsecond2,
  asecond3,
  asecond2,
  Ymeter3,
  Ymeter2,
  kmeter3,
  kmeter2,
  Mmeter3,
  Mmeter2,
  zmeter3,
  zmeter2,
  Emeter3,
  Emeter2,
  ymeter3,
  ymeter2,
  Tmeter3,
  Tmeter2,
  umeter3,
  umeter2,
  pmeter3,
  pmeter2,
  nmeter3,
  nmeter2,
  mmeter3,
  mmeter2,
  fmeter3,
  fmeter2,
  Gmeter3,
  Gmeter2,
  Pmeter3,
  Pmeter2,
  Zmeter3,
  Zmeter2,
  ameter3,
  ameter2,
  Ymetre3,
  Ymetre2,
  kmetre3,
  kmetre2,
  Mmetre3,
  Mmetre2,
  zmetre3,
  zmetre2,
  Emetre3,
  Emetre2,
  ymetre3,
  ymetre2,
  Tmetre3,
  Tmetre2,
  umetre3,
  umetre2,
  pmetre3,
  pmetre2,
  nmetre3,
  nmetre2,
  mmetre3,
  mmetre2,
  fmetre3,
  fmetre2,
  Gmetre3,
  Gmetre2,
  Pmetre3,
  Pmetre2,
  Zmetre3,
  Zmetre2,
  ametre3,
  ametre2,
  katal3,
  katal2,
  sievert3,
  sievert2,
  gray3,
  gray2,
  becquerel3,
  becquerel2,
  lux3,
  lux2,
  lumen3,
  lumen2,
  henry3,
  henry2,
  tesla3,
  tesla2,
  weber3,
  weber2,
  siemens3,
  siemens2,
  ohm3,
  ohm2,
  farad3,
  farad2,
  volt3,
  volt2,
  coulomb3,
  coulomb2,
  watt3,
  watt2,
  joule3,
  joule2,
  pascal3,
  pascal2,
  newton3,
  newton2,
  hertz3,
  hertz2,
  steradian3,
  steradian2,
  radian3,
  radian2,
  molar3,
  molar2,
  gramme3,
  gramme2,
  gram3,
  gram2,
  kilogramme3,
  kilogramme2,
  candle3,
  candle2,
  mol3,
  mol2,
  mole3,
  mole2,
  kelvin3,
  kelvin2,
  ampere3,
  ampere2,
  amp3,
  amp2,
  second3,
  second2,
  kilogram3,
  kilogram2,
  meter3,
  meter2,
  metre3,
  metre2,
]

# Current list from http://physics.nist.gov/cuu/Units/units.html, far from complete
additional_units = [
  pascal * second, newton * metre, watt / metre ** 2, joule / kelvin,
  joule / (kilogram * kelvin), joule / kilogram, watt / (metre * kelvin),
  joule / metre ** 3, volt / metre ** 3, coulomb / metre ** 3, coulomb / metre ** 2,
  farad / metre, henry / metre, joule / mole, joule / (mole * kelvin),
  coulomb / kilogram, gray / second, katal / metre ** 3,
  # We don't want liter/litre to be used as a standard unit for display, so we
  # put it here instead of in the standard units
  aliter, liter, cliter, Zliter, Pliter, dliter, Gliter, fliter, hliter, daliter, mliter, nliter, pliter, uliter,
  Tliter, yliter, Eliter, zliter, Mliter, kliter, Yliter, alitre, litre, clitre, Zlitre, Plitre, dlitre, Glitre, flitre,
  hlitre, dalitre, mlitre, nlitre, plitre, ulitre, Tlitre, ylitre, Elitre, zlitre, Mlitre, klitre, Ylitre,
]

all_units = [
  Ylitre,
  klitre,
  Mlitre,
  zlitre,
  Elitre,
  ylitre,
  Tlitre,
  ulitre,
  plitre,
  nlitre,
  mlitre,
  dalitre,
  hlitre,
  flitre,
  Glitre,
  dlitre,
  Plitre,
  Zlitre,
  clitre,
  litre,
  alitre,
  litre,
  Yliter,
  kliter,
  Mliter,
  zliter,
  Eliter,
  yliter,
  Tliter,
  uliter,
  pliter,
  nliter,
  mliter,
  daliter,
  hliter,
  fliter,
  Gliter,
  dliter,
  Pliter,
  Zliter,
  cliter,
  liter,
  aliter,
  liter,
  Ykatal3,
  Ykatal2,
  kkatal3,
  kkatal2,
  Mkatal3,
  Mkatal2,
  zkatal3,
  zkatal2,
  Ekatal3,
  Ekatal2,
  ykatal3,
  ykatal2,
  Tkatal3,
  Tkatal2,
  ukatal3,
  ukatal2,
  pkatal3,
  pkatal2,
  nkatal3,
  nkatal2,
  mkatal3,
  mkatal2,
  dakatal3,
  dakatal2,
  hkatal3,
  hkatal2,
  fkatal3,
  fkatal2,
  Gkatal3,
  Gkatal2,
  dkatal3,
  dkatal2,
  Pkatal3,
  Pkatal2,
  Zkatal3,
  Zkatal2,
  ckatal3,
  ckatal2,
  akatal3,
  akatal2,
  Ysievert3,
  Ysievert2,
  ksievert3,
  ksievert2,
  Msievert3,
  Msievert2,
  zsievert3,
  zsievert2,
  Esievert3,
  Esievert2,
  ysievert3,
  ysievert2,
  Tsievert3,
  Tsievert2,
  usievert3,
  usievert2,
  psievert3,
  psievert2,
  nsievert3,
  nsievert2,
  msievert3,
  msievert2,
  dasievert3,
  dasievert2,
  hsievert3,
  hsievert2,
  fsievert3,
  fsievert2,
  Gsievert3,
  Gsievert2,
  dsievert3,
  dsievert2,
  Psievert3,
  Psievert2,
  Zsievert3,
  Zsievert2,
  csievert3,
  csievert2,
  asievert3,
  asievert2,
  Ygray3,
  Ygray2,
  kgray3,
  kgray2,
  Mgray3,
  Mgray2,
  zgray3,
  zgray2,
  Egray3,
  Egray2,
  ygray3,
  ygray2,
  Tgray3,
  Tgray2,
  ugray3,
  ugray2,
  pgray3,
  pgray2,
  ngray3,
  ngray2,
  mgray3,
  mgray2,
  dagray3,
  dagray2,
  hgray3,
  hgray2,
  fgray3,
  fgray2,
  Ggray3,
  Ggray2,
  dgray3,
  dgray2,
  Pgray3,
  Pgray2,
  Zgray3,
  Zgray2,
  cgray3,
  cgray2,
  agray3,
  agray2,
  Ybecquerel3,
  Ybecquerel2,
  kbecquerel3,
  kbecquerel2,
  Mbecquerel3,
  Mbecquerel2,
  zbecquerel3,
  zbecquerel2,
  Ebecquerel3,
  Ebecquerel2,
  ybecquerel3,
  ybecquerel2,
  Tbecquerel3,
  Tbecquerel2,
  ubecquerel3,
  ubecquerel2,
  pbecquerel3,
  pbecquerel2,
  nbecquerel3,
  nbecquerel2,
  mbecquerel3,
  mbecquerel2,
  dabecquerel3,
  dabecquerel2,
  hbecquerel3,
  hbecquerel2,
  fbecquerel3,
  fbecquerel2,
  Gbecquerel3,
  Gbecquerel2,
  dbecquerel3,
  dbecquerel2,
  Pbecquerel3,
  Pbecquerel2,
  Zbecquerel3,
  Zbecquerel2,
  cbecquerel3,
  cbecquerel2,
  abecquerel3,
  abecquerel2,
  Ylux3,
  Ylux2,
  klux3,
  klux2,
  Mlux3,
  Mlux2,
  zlux3,
  zlux2,
  Elux3,
  Elux2,
  ylux3,
  ylux2,
  Tlux3,
  Tlux2,
  ulux3,
  ulux2,
  plux3,
  plux2,
  nlux3,
  nlux2,
  mlux3,
  mlux2,
  dalux3,
  dalux2,
  hlux3,
  hlux2,
  flux3,
  flux2,
  Glux3,
  Glux2,
  dlux3,
  dlux2,
  Plux3,
  Plux2,
  Zlux3,
  Zlux2,
  clux3,
  clux2,
  alux3,
  alux2,
  Ylumen3,
  Ylumen2,
  klumen3,
  klumen2,
  Mlumen3,
  Mlumen2,
  zlumen3,
  zlumen2,
  Elumen3,
  Elumen2,
  ylumen3,
  ylumen2,
  Tlumen3,
  Tlumen2,
  ulumen3,
  ulumen2,
  plumen3,
  plumen2,
  nlumen3,
  nlumen2,
  mlumen3,
  mlumen2,
  dalumen3,
  dalumen2,
  hlumen3,
  hlumen2,
  flumen3,
  flumen2,
  Glumen3,
  Glumen2,
  dlumen3,
  dlumen2,
  Plumen3,
  Plumen2,
  Zlumen3,
  Zlumen2,
  clumen3,
  clumen2,
  alumen3,
  alumen2,
  Yhenry3,
  Yhenry2,
  khenry3,
  khenry2,
  Mhenry3,
  Mhenry2,
  zhenry3,
  zhenry2,
  Ehenry3,
  Ehenry2,
  yhenry3,
  yhenry2,
  Thenry3,
  Thenry2,
  uhenry3,
  uhenry2,
  phenry3,
  phenry2,
  nhenry3,
  nhenry2,
  mhenry3,
  mhenry2,
  dahenry3,
  dahenry2,
  hhenry3,
  hhenry2,
  fhenry3,
  fhenry2,
  Ghenry3,
  Ghenry2,
  dhenry3,
  dhenry2,
  Phenry3,
  Phenry2,
  Zhenry3,
  Zhenry2,
  chenry3,
  chenry2,
  ahenry3,
  ahenry2,
  Ytesla3,
  Ytesla2,
  ktesla3,
  ktesla2,
  Mtesla3,
  Mtesla2,
  ztesla3,
  ztesla2,
  Etesla3,
  Etesla2,
  ytesla3,
  ytesla2,
  Ttesla3,
  Ttesla2,
  utesla3,
  utesla2,
  ptesla3,
  ptesla2,
  ntesla3,
  ntesla2,
  mtesla3,
  mtesla2,
  datesla3,
  datesla2,
  htesla3,
  htesla2,
  ftesla3,
  ftesla2,
  Gtesla3,
  Gtesla2,
  dtesla3,
  dtesla2,
  Ptesla3,
  Ptesla2,
  Ztesla3,
  Ztesla2,
  ctesla3,
  ctesla2,
  atesla3,
  atesla2,
  Yweber3,
  Yweber2,
  kweber3,
  kweber2,
  Mweber3,
  Mweber2,
  zweber3,
  zweber2,
  Eweber3,
  Eweber2,
  yweber3,
  yweber2,
  Tweber3,
  Tweber2,
  uweber3,
  uweber2,
  pweber3,
  pweber2,
  nweber3,
  nweber2,
  mweber3,
  mweber2,
  daweber3,
  daweber2,
  hweber3,
  hweber2,
  fweber3,
  fweber2,
  Gweber3,
  Gweber2,
  dweber3,
  dweber2,
  Pweber3,
  Pweber2,
  Zweber3,
  Zweber2,
  cweber3,
  cweber2,
  aweber3,
  aweber2,
  Ysiemens3,
  Ysiemens2,
  ksiemens3,
  ksiemens2,
  Msiemens3,
  Msiemens2,
  zsiemens3,
  zsiemens2,
  Esiemens3,
  Esiemens2,
  ysiemens3,
  ysiemens2,
  Tsiemens3,
  Tsiemens2,
  usiemens3,
  usiemens2,
  psiemens3,
  psiemens2,
  nsiemens3,
  nsiemens2,
  msiemens3,
  msiemens2,
  dasiemens3,
  dasiemens2,
  hsiemens3,
  hsiemens2,
  fsiemens3,
  fsiemens2,
  Gsiemens3,
  Gsiemens2,
  dsiemens3,
  dsiemens2,
  Psiemens3,
  Psiemens2,
  Zsiemens3,
  Zsiemens2,
  csiemens3,
  csiemens2,
  asiemens3,
  asiemens2,
  Yohm3,
  Yohm2,
  kohm3,
  kohm2,
  Mohm3,
  Mohm2,
  zohm3,
  zohm2,
  Eohm3,
  Eohm2,
  yohm3,
  yohm2,
  Tohm3,
  Tohm2,
  uohm3,
  uohm2,
  pohm3,
  pohm2,
  nohm3,
  nohm2,
  mohm3,
  mohm2,
  daohm3,
  daohm2,
  hohm3,
  hohm2,
  fohm3,
  fohm2,
  Gohm3,
  Gohm2,
  dohm3,
  dohm2,
  Pohm3,
  Pohm2,
  Zohm3,
  Zohm2,
  cohm3,
  cohm2,
  aohm3,
  aohm2,
  Yfarad3,
  Yfarad2,
  kfarad3,
  kfarad2,
  Mfarad3,
  Mfarad2,
  zfarad3,
  zfarad2,
  Efarad3,
  Efarad2,
  yfarad3,
  yfarad2,
  Tfarad3,
  Tfarad2,
  ufarad3,
  ufarad2,
  pfarad3,
  pfarad2,
  nfarad3,
  nfarad2,
  mfarad3,
  mfarad2,
  dafarad3,
  dafarad2,
  hfarad3,
  hfarad2,
  ffarad3,
  ffarad2,
  Gfarad3,
  Gfarad2,
  dfarad3,
  dfarad2,
  Pfarad3,
  Pfarad2,
  Zfarad3,
  Zfarad2,
  cfarad3,
  cfarad2,
  afarad3,
  afarad2,
  Yvolt3,
  Yvolt2,
  kvolt3,
  kvolt2,
  Mvolt3,
  Mvolt2,
  zvolt3,
  zvolt2,
  Evolt3,
  Evolt2,
  yvolt3,
  yvolt2,
  Tvolt3,
  Tvolt2,
  uvolt3,
  uvolt2,
  pvolt3,
  pvolt2,
  nvolt3,
  nvolt2,
  mvolt3,
  mvolt2,
  davolt3,
  davolt2,
  hvolt3,
  hvolt2,
  fvolt3,
  fvolt2,
  Gvolt3,
  Gvolt2,
  dvolt3,
  dvolt2,
  Pvolt3,
  Pvolt2,
  Zvolt3,
  Zvolt2,
  cvolt3,
  cvolt2,
  avolt3,
  avolt2,
  Ycoulomb3,
  Ycoulomb2,
  kcoulomb3,
  kcoulomb2,
  Mcoulomb3,
  Mcoulomb2,
  zcoulomb3,
  zcoulomb2,
  Ecoulomb3,
  Ecoulomb2,
  ycoulomb3,
  ycoulomb2,
  Tcoulomb3,
  Tcoulomb2,
  ucoulomb3,
  ucoulomb2,
  pcoulomb3,
  pcoulomb2,
  ncoulomb3,
  ncoulomb2,
  mcoulomb3,
  mcoulomb2,
  dacoulomb3,
  dacoulomb2,
  hcoulomb3,
  hcoulomb2,
  fcoulomb3,
  fcoulomb2,
  Gcoulomb3,
  Gcoulomb2,
  dcoulomb3,
  dcoulomb2,
  Pcoulomb3,
  Pcoulomb2,
  Zcoulomb3,
  Zcoulomb2,
  ccoulomb3,
  ccoulomb2,
  acoulomb3,
  acoulomb2,
  Ywatt3,
  Ywatt2,
  kwatt3,
  kwatt2,
  Mwatt3,
  Mwatt2,
  zwatt3,
  zwatt2,
  Ewatt3,
  Ewatt2,
  ywatt3,
  ywatt2,
  Twatt3,
  Twatt2,
  uwatt3,
  uwatt2,
  pwatt3,
  pwatt2,
  nwatt3,
  nwatt2,
  mwatt3,
  mwatt2,
  dawatt3,
  dawatt2,
  hwatt3,
  hwatt2,
  fwatt3,
  fwatt2,
  Gwatt3,
  Gwatt2,
  dwatt3,
  dwatt2,
  Pwatt3,
  Pwatt2,
  Zwatt3,
  Zwatt2,
  cwatt3,
  cwatt2,
  awatt3,
  awatt2,
  Yjoule3,
  Yjoule2,
  kjoule3,
  kjoule2,
  Mjoule3,
  Mjoule2,
  zjoule3,
  zjoule2,
  Ejoule3,
  Ejoule2,
  yjoule3,
  yjoule2,
  Tjoule3,
  Tjoule2,
  ujoule3,
  ujoule2,
  pjoule3,
  pjoule2,
  njoule3,
  njoule2,
  mjoule3,
  mjoule2,
  dajoule3,
  dajoule2,
  hjoule3,
  hjoule2,
  fjoule3,
  fjoule2,
  Gjoule3,
  Gjoule2,
  djoule3,
  djoule2,
  Pjoule3,
  Pjoule2,
  Zjoule3,
  Zjoule2,
  cjoule3,
  cjoule2,
  ajoule3,
  ajoule2,
  Ypascal3,
  Ypascal2,
  kpascal3,
  kpascal2,
  Mpascal3,
  Mpascal2,
  zpascal3,
  zpascal2,
  Epascal3,
  Epascal2,
  ypascal3,
  ypascal2,
  Tpascal3,
  Tpascal2,
  upascal3,
  upascal2,
  ppascal3,
  ppascal2,
  npascal3,
  npascal2,
  mpascal3,
  mpascal2,
  dapascal3,
  dapascal2,
  hpascal3,
  hpascal2,
  fpascal3,
  fpascal2,
  Gpascal3,
  Gpascal2,
  dpascal3,
  dpascal2,
  Ppascal3,
  Ppascal2,
  Zpascal3,
  Zpascal2,
  cpascal3,
  cpascal2,
  apascal3,
  apascal2,
  Ynewton3,
  Ynewton2,
  knewton3,
  knewton2,
  Mnewton3,
  Mnewton2,
  znewton3,
  znewton2,
  Enewton3,
  Enewton2,
  ynewton3,
  ynewton2,
  Tnewton3,
  Tnewton2,
  unewton3,
  unewton2,
  pnewton3,
  pnewton2,
  nnewton3,
  nnewton2,
  mnewton3,
  mnewton2,
  danewton3,
  danewton2,
  hnewton3,
  hnewton2,
  fnewton3,
  fnewton2,
  Gnewton3,
  Gnewton2,
  dnewton3,
  dnewton2,
  Pnewton3,
  Pnewton2,
  Znewton3,
  Znewton2,
  cnewton3,
  cnewton2,
  anewton3,
  anewton2,
  Yhertz3,
  Yhertz2,
  khertz3,
  khertz2,
  Mhertz3,
  Mhertz2,
  zhertz3,
  zhertz2,
  Ehertz3,
  Ehertz2,
  yhertz3,
  yhertz2,
  Thertz3,
  Thertz2,
  uhertz3,
  uhertz2,
  phertz3,
  phertz2,
  nhertz3,
  nhertz2,
  mhertz3,
  mhertz2,
  dahertz3,
  dahertz2,
  hhertz3,
  hhertz2,
  fhertz3,
  fhertz2,
  Ghertz3,
  Ghertz2,
  dhertz3,
  dhertz2,
  Phertz3,
  Phertz2,
  Zhertz3,
  Zhertz2,
  chertz3,
  chertz2,
  ahertz3,
  ahertz2,
  Ysteradian3,
  Ysteradian2,
  ksteradian3,
  ksteradian2,
  Msteradian3,
  Msteradian2,
  zsteradian3,
  zsteradian2,
  Esteradian3,
  Esteradian2,
  ysteradian3,
  ysteradian2,
  Tsteradian3,
  Tsteradian2,
  usteradian3,
  usteradian2,
  psteradian3,
  psteradian2,
  nsteradian3,
  nsteradian2,
  msteradian3,
  msteradian2,
  dasteradian3,
  dasteradian2,
  hsteradian3,
  hsteradian2,
  fsteradian3,
  fsteradian2,
  Gsteradian3,
  Gsteradian2,
  dsteradian3,
  dsteradian2,
  Psteradian3,
  Psteradian2,
  Zsteradian3,
  Zsteradian2,
  csteradian3,
  csteradian2,
  asteradian3,
  asteradian2,
  Yradian3,
  Yradian2,
  kradian3,
  kradian2,
  Mradian3,
  Mradian2,
  zradian3,
  zradian2,
  Eradian3,
  Eradian2,
  yradian3,
  yradian2,
  Tradian3,
  Tradian2,
  uradian3,
  uradian2,
  pradian3,
  pradian2,
  nradian3,
  nradian2,
  mradian3,
  mradian2,
  daradian3,
  daradian2,
  hradian3,
  hradian2,
  fradian3,
  fradian2,
  Gradian3,
  Gradian2,
  dradian3,
  dradian2,
  Pradian3,
  Pradian2,
  Zradian3,
  Zradian2,
  cradian3,
  cradian2,
  aradian3,
  aradian2,
  Ymolar3,
  Ymolar2,
  kmolar3,
  kmolar2,
  Mmolar3,
  Mmolar2,
  zmolar3,
  zmolar2,
  Emolar3,
  Emolar2,
  ymolar3,
  ymolar2,
  Tmolar3,
  Tmolar2,
  umolar3,
  umolar2,
  pmolar3,
  pmolar2,
  nmolar3,
  nmolar2,
  mmolar3,
  mmolar2,
  damolar3,
  damolar2,
  hmolar3,
  hmolar2,
  fmolar3,
  fmolar2,
  Gmolar3,
  Gmolar2,
  dmolar3,
  dmolar2,
  Pmolar3,
  Pmolar2,
  Zmolar3,
  Zmolar2,
  cmolar3,
  cmolar2,
  amolar3,
  amolar2,
  Ygramme3,
  Ygramme2,
  kgramme3,
  kgramme2,
  Mgramme3,
  Mgramme2,
  zgramme3,
  zgramme2,
  Egramme3,
  Egramme2,
  ygramme3,
  ygramme2,
  Tgramme3,
  Tgramme2,
  ugramme3,
  ugramme2,
  pgramme3,
  pgramme2,
  ngramme3,
  ngramme2,
  mgramme3,
  mgramme2,
  dagramme3,
  dagramme2,
  hgramme3,
  hgramme2,
  fgramme3,
  fgramme2,
  Ggramme3,
  Ggramme2,
  dgramme3,
  dgramme2,
  Pgramme3,
  Pgramme2,
  Zgramme3,
  Zgramme2,
  cgramme3,
  cgramme2,
  agramme3,
  agramme2,
  Ygram3,
  Ygram2,
  kgram3,
  kgram2,
  Mgram3,
  Mgram2,
  zgram3,
  zgram2,
  Egram3,
  Egram2,
  ygram3,
  ygram2,
  Tgram3,
  Tgram2,
  ugram3,
  ugram2,
  pgram3,
  pgram2,
  ngram3,
  ngram2,
  mgram3,
  mgram2,
  dagram3,
  dagram2,
  hgram3,
  hgram2,
  fgram3,
  fgram2,
  Ggram3,
  Ggram2,
  dgram3,
  dgram2,
  Pgram3,
  Pgram2,
  Zgram3,
  Zgram2,
  cgram3,
  cgram2,
  agram3,
  agram2,
  Ycandle3,
  Ycandle2,
  kcandle3,
  kcandle2,
  Mcandle3,
  Mcandle2,
  zcandle3,
  zcandle2,
  Ecandle3,
  Ecandle2,
  ycandle3,
  ycandle2,
  Tcandle3,
  Tcandle2,
  ucandle3,
  ucandle2,
  pcandle3,
  pcandle2,
  ncandle3,
  ncandle2,
  mcandle3,
  mcandle2,
  dacandle3,
  dacandle2,
  hcandle3,
  hcandle2,
  fcandle3,
  fcandle2,
  Gcandle3,
  Gcandle2,
  dcandle3,
  dcandle2,
  Pcandle3,
  Pcandle2,
  Zcandle3,
  Zcandle2,
  ccandle3,
  ccandle2,
  acandle3,
  acandle2,
  Ymol3,
  Ymol2,
  kmol3,
  kmol2,
  Mmol3,
  Mmol2,
  zmol3,
  zmol2,
  Emol3,
  Emol2,
  ymol3,
  ymol2,
  Tmol3,
  Tmol2,
  umol3,
  umol2,
  pmol3,
  pmol2,
  nmol3,
  nmol2,
  mmol3,
  mmol2,
  damol3,
  damol2,
  hmol3,
  hmol2,
  fmol3,
  fmol2,
  Gmol3,
  Gmol2,
  dmol3,
  dmol2,
  Pmol3,
  Pmol2,
  Zmol3,
  Zmol2,
  cmol3,
  cmol2,
  amol3,
  amol2,
  Ymole3,
  Ymole2,
  kmole3,
  kmole2,
  Mmole3,
  Mmole2,
  zmole3,
  zmole2,
  Emole3,
  Emole2,
  ymole3,
  ymole2,
  Tmole3,
  Tmole2,
  umole3,
  umole2,
  pmole3,
  pmole2,
  nmole3,
  nmole2,
  mmole3,
  mmole2,
  damole3,
  damole2,
  hmole3,
  hmole2,
  fmole3,
  fmole2,
  Gmole3,
  Gmole2,
  dmole3,
  dmole2,
  Pmole3,
  Pmole2,
  Zmole3,
  Zmole2,
  cmole3,
  cmole2,
  amole3,
  amole2,
  Yampere3,
  Yampere2,
  kampere3,
  kampere2,
  Mampere3,
  Mampere2,
  zampere3,
  zampere2,
  Eampere3,
  Eampere2,
  yampere3,
  yampere2,
  Tampere3,
  Tampere2,
  uampere3,
  uampere2,
  pampere3,
  pampere2,
  nampere3,
  nampere2,
  mampere3,
  mampere2,
  daampere3,
  daampere2,
  hampere3,
  hampere2,
  fampere3,
  fampere2,
  Gampere3,
  Gampere2,
  dampere3,
  dampere2,
  Pampere3,
  Pampere2,
  Zampere3,
  Zampere2,
  campere3,
  campere2,
  aampere3,
  aampere2,
  Yamp3,
  Yamp2,
  kamp3,
  kamp2,
  Mamp3,
  Mamp2,
  zamp3,
  zamp2,
  Eamp3,
  Eamp2,
  yamp3,
  yamp2,
  Tamp3,
  Tamp2,
  uamp3,
  uamp2,
  pamp3,
  pamp2,
  namp3,
  namp2,
  mamp3,
  mamp2,
  daamp3,
  daamp2,
  hamp3,
  hamp2,
  famp3,
  famp2,
  Gamp3,
  Gamp2,
  damp3,
  damp2,
  Pamp3,
  Pamp2,
  Zamp3,
  Zamp2,
  camp3,
  camp2,
  aamp3,
  aamp2,
  Ysecond3,
  Ysecond2,
  ksecond3,
  ksecond2,
  Msecond3,
  Msecond2,
  zsecond3,
  zsecond2,
  Esecond3,
  Esecond2,
  ysecond3,
  ysecond2,
  Tsecond3,
  Tsecond2,
  usecond3,
  usecond2,
  psecond3,
  psecond2,
  nsecond3,
  nsecond2,
  msecond3,
  msecond2,
  dasecond3,
  dasecond2,
  hsecond3,
  hsecond2,
  fsecond3,
  fsecond2,
  Gsecond3,
  Gsecond2,
  dsecond3,
  dsecond2,
  Psecond3,
  Psecond2,
  Zsecond3,
  Zsecond2,
  csecond3,
  csecond2,
  asecond3,
  asecond2,
  Ymeter3,
  Ymeter2,
  kmeter3,
  kmeter2,
  Mmeter3,
  Mmeter2,
  zmeter3,
  zmeter2,
  Emeter3,
  Emeter2,
  ymeter3,
  ymeter2,
  Tmeter3,
  Tmeter2,
  umeter3,
  umeter2,
  pmeter3,
  pmeter2,
  nmeter3,
  nmeter2,
  mmeter3,
  mmeter2,
  dameter3,
  dameter2,
  hmeter3,
  hmeter2,
  fmeter3,
  fmeter2,
  Gmeter3,
  Gmeter2,
  dmeter3,
  dmeter2,
  Pmeter3,
  Pmeter2,
  Zmeter3,
  Zmeter2,
  cmeter3,
  cmeter2,
  ameter3,
  ameter2,
  Ymetre3,
  Ymetre2,
  kmetre3,
  kmetre2,
  Mmetre3,
  Mmetre2,
  zmetre3,
  zmetre2,
  Emetre3,
  Emetre2,
  ymetre3,
  ymetre2,
  Tmetre3,
  Tmetre2,
  umetre3,
  umetre2,
  pmetre3,
  pmetre2,
  nmetre3,
  nmetre2,
  mmetre3,
  mmetre2,
  dametre3,
  dametre2,
  hmetre3,
  hmetre2,
  fmetre3,
  fmetre2,
  Gmetre3,
  Gmetre2,
  dmetre3,
  dmetre2,
  Pmetre3,
  Pmetre2,
  Zmetre3,
  Zmetre2,
  cmetre3,
  cmetre2,
  ametre3,
  ametre2,
  katal3,
  katal2,
  sievert3,
  sievert2,
  gray3,
  gray2,
  becquerel3,
  becquerel2,
  lux3,
  lux2,
  lumen3,
  lumen2,
  henry3,
  henry2,
  tesla3,
  tesla2,
  weber3,
  weber2,
  siemens3,
  siemens2,
  ohm3,
  ohm2,
  farad3,
  farad2,
  volt3,
  volt2,
  coulomb3,
  coulomb2,
  watt3,
  watt2,
  joule3,
  joule2,
  pascal3,
  pascal2,
  newton3,
  newton2,
  hertz3,
  hertz2,
  steradian3,
  steradian2,
  radian3,
  radian2,
  molar3,
  molar2,
  gramme3,
  gramme2,
  gram3,
  gram2,
  kilogramme3,
  kilogramme2,
  candle3,
  candle2,
  mol3,
  mol2,
  mole3,
  mole2,
  kelvin3,
  kelvin2,
  ampere3,
  ampere2,
  amp3,
  amp2,
  second3,
  second2,
  kilogram3,
  kilogram2,
  meter3,
  meter2,
  metre3,
  metre2,
  Ykatal,
  kkatal,
  Mkatal,
  zkatal,
  Ekatal,
  ykatal,
  Tkatal,
  ukatal,
  pkatal,
  nkatal,
  mkatal,
  dakatal,
  hkatal,
  fkatal,
  Gkatal,
  dkatal,
  Pkatal,
  Zkatal,
  ckatal,
  akatal,
  Ysievert,
  ksievert,
  Msievert,
  zsievert,
  Esievert,
  ysievert,
  Tsievert,
  usievert,
  psievert,
  nsievert,
  msievert,
  dasievert,
  hsievert,
  fsievert,
  Gsievert,
  dsievert,
  Psievert,
  Zsievert,
  csievert,
  asievert,
  Ygray,
  kgray,
  Mgray,
  zgray,
  Egray,
  ygray,
  Tgray,
  ugray,
  pgray,
  ngray,
  mgray,
  dagray,
  hgray,
  fgray,
  Ggray,
  dgray,
  Pgray,
  Zgray,
  cgray,
  agray,
  Ybecquerel,
  kbecquerel,
  Mbecquerel,
  zbecquerel,
  Ebecquerel,
  ybecquerel,
  Tbecquerel,
  ubecquerel,
  pbecquerel,
  nbecquerel,
  mbecquerel,
  dabecquerel,
  hbecquerel,
  fbecquerel,
  Gbecquerel,
  dbecquerel,
  Pbecquerel,
  Zbecquerel,
  cbecquerel,
  abecquerel,
  Ylux,
  klux,
  Mlux,
  zlux,
  Elux,
  ylux,
  Tlux,
  ulux,
  plux,
  nlux,
  mlux,
  dalux,
  hlux,
  flux,
  Glux,
  dlux,
  Plux,
  Zlux,
  clux,
  alux,
  Ylumen,
  klumen,
  Mlumen,
  zlumen,
  Elumen,
  ylumen,
  Tlumen,
  ulumen,
  plumen,
  nlumen,
  mlumen,
  dalumen,
  hlumen,
  flumen,
  Glumen,
  dlumen,
  Plumen,
  Zlumen,
  clumen,
  alumen,
  Yhenry,
  khenry,
  Mhenry,
  zhenry,
  Ehenry,
  yhenry,
  Thenry,
  uhenry,
  phenry,
  nhenry,
  mhenry,
  dahenry,
  hhenry,
  fhenry,
  Ghenry,
  dhenry,
  Phenry,
  Zhenry,
  chenry,
  ahenry,
  Ytesla,
  ktesla,
  Mtesla,
  ztesla,
  Etesla,
  ytesla,
  Ttesla,
  utesla,
  ptesla,
  ntesla,
  mtesla,
  datesla,
  htesla,
  ftesla,
  Gtesla,
  dtesla,
  Ptesla,
  Ztesla,
  ctesla,
  atesla,
  Yweber,
  kweber,
  Mweber,
  zweber,
  Eweber,
  yweber,
  Tweber,
  uweber,
  pweber,
  nweber,
  mweber,
  daweber,
  hweber,
  fweber,
  Gweber,
  dweber,
  Pweber,
  Zweber,
  cweber,
  aweber,
  Ysiemens,
  ksiemens,
  Msiemens,
  zsiemens,
  Esiemens,
  ysiemens,
  Tsiemens,
  usiemens,
  psiemens,
  nsiemens,
  msiemens,
  dasiemens,
  hsiemens,
  fsiemens,
  Gsiemens,
  dsiemens,
  Psiemens,
  Zsiemens,
  csiemens,
  asiemens,
  Yohm,
  kohm,
  Mohm,
  zohm,
  Eohm,
  yohm,
  Tohm,
  uohm,
  pohm,
  nohm,
  mohm,
  daohm,
  hohm,
  fohm,
  Gohm,
  dohm,
  Pohm,
  Zohm,
  cohm,
  aohm,
  Yfarad,
  kfarad,
  Mfarad,
  zfarad,
  Efarad,
  yfarad,
  Tfarad,
  ufarad,
  pfarad,
  nfarad,
  mfarad,
  dafarad,
  hfarad,
  ffarad,
  Gfarad,
  dfarad,
  Pfarad,
  Zfarad,
  cfarad,
  afarad,
  Yvolt,
  kvolt,
  Mvolt,
  zvolt,
  Evolt,
  yvolt,
  Tvolt,
  uvolt,
  pvolt,
  nvolt,
  mvolt,
  davolt,
  hvolt,
  fvolt,
  Gvolt,
  dvolt,
  Pvolt,
  Zvolt,
  cvolt,
  avolt,
  Ycoulomb,
  kcoulomb,
  Mcoulomb,
  zcoulomb,
  Ecoulomb,
  ycoulomb,
  Tcoulomb,
  ucoulomb,
  pcoulomb,
  ncoulomb,
  mcoulomb,
  dacoulomb,
  hcoulomb,
  fcoulomb,
  Gcoulomb,
  dcoulomb,
  Pcoulomb,
  Zcoulomb,
  ccoulomb,
  acoulomb,
  Ywatt,
  kwatt,
  Mwatt,
  zwatt,
  Ewatt,
  ywatt,
  Twatt,
  uwatt,
  pwatt,
  nwatt,
  mwatt,
  dawatt,
  hwatt,
  fwatt,
  Gwatt,
  dwatt,
  Pwatt,
  Zwatt,
  cwatt,
  awatt,
  Yjoule,
  kjoule,
  Mjoule,
  zjoule,
  Ejoule,
  yjoule,
  Tjoule,
  ujoule,
  pjoule,
  njoule,
  mjoule,
  dajoule,
  hjoule,
  fjoule,
  Gjoule,
  djoule,
  Pjoule,
  Zjoule,
  cjoule,
  ajoule,
  Ypascal,
  kpascal,
  Mpascal,
  zpascal,
  Epascal,
  ypascal,
  Tpascal,
  upascal,
  ppascal,
  npascal,
  mpascal,
  dapascal,
  hpascal,
  fpascal,
  Gpascal,
  dpascal,
  Ppascal,
  Zpascal,
  cpascal,
  apascal,
  Ynewton,
  knewton,
  Mnewton,
  znewton,
  Enewton,
  ynewton,
  Tnewton,
  unewton,
  pnewton,
  nnewton,
  mnewton,
  danewton,
  hnewton,
  fnewton,
  Gnewton,
  dnewton,
  Pnewton,
  Znewton,
  cnewton,
  anewton,
  Yhertz,
  khertz,
  Mhertz,
  zhertz,
  Ehertz,
  yhertz,
  Thertz,
  uhertz,
  phertz,
  nhertz,
  mhertz,
  dahertz,
  hhertz,
  fhertz,
  Ghertz,
  dhertz,
  Phertz,
  Zhertz,
  chertz,
  ahertz,
  Ysteradian,
  ksteradian,
  Msteradian,
  zsteradian,
  Esteradian,
  ysteradian,
  Tsteradian,
  usteradian,
  psteradian,
  nsteradian,
  msteradian,
  dasteradian,
  hsteradian,
  fsteradian,
  Gsteradian,
  dsteradian,
  Psteradian,
  Zsteradian,
  csteradian,
  asteradian,
  Yradian,
  kradian,
  Mradian,
  zradian,
  Eradian,
  yradian,
  Tradian,
  uradian,
  pradian,
  nradian,
  mradian,
  daradian,
  hradian,
  fradian,
  Gradian,
  dradian,
  Pradian,
  Zradian,
  cradian,
  aradian,
  Ymolar,
  kmolar,
  Mmolar,
  zmolar,
  Emolar,
  ymolar,
  Tmolar,
  umolar,
  pmolar,
  nmolar,
  mmolar,
  damolar,
  hmolar,
  fmolar,
  Gmolar,
  dmolar,
  Pmolar,
  Zmolar,
  cmolar,
  amolar,
  Ygramme,
  kgramme,
  Mgramme,
  zgramme,
  Egramme,
  ygramme,
  Tgramme,
  ugramme,
  pgramme,
  ngramme,
  mgramme,
  dagramme,
  hgramme,
  fgramme,
  Ggramme,
  dgramme,
  Pgramme,
  Zgramme,
  cgramme,
  agramme,
  Ygram,
  kgram,
  Mgram,
  zgram,
  Egram,
  ygram,
  Tgram,
  ugram,
  pgram,
  ngram,
  mgram,
  dagram,
  hgram,
  fgram,
  Ggram,
  dgram,
  Pgram,
  Zgram,
  cgram,
  agram,
  Ycandle,
  kcandle,
  Mcandle,
  zcandle,
  Ecandle,
  ycandle,
  Tcandle,
  ucandle,
  pcandle,
  ncandle,
  mcandle,
  dacandle,
  hcandle,
  fcandle,
  Gcandle,
  dcandle,
  Pcandle,
  Zcandle,
  ccandle,
  acandle,
  Ymol,
  kmol,
  Mmol,
  zmol,
  Emol,
  ymol,
  Tmol,
  umol,
  pmol,
  nmol,
  mmol,
  damol,
  hmol,
  fmol,
  Gmol,
  dmol,
  Pmol,
  Zmol,
  cmol,
  amol,
  Ymole,
  kmole,
  Mmole,
  zmole,
  Emole,
  ymole,
  Tmole,
  umole,
  pmole,
  nmole,
  mmole,
  damole,
  hmole,
  fmole,
  Gmole,
  dmole,
  Pmole,
  Zmole,
  cmole,
  amole,
  Yampere,
  kampere,
  Mampere,
  zampere,
  Eampere,
  yampere,
  Tampere,
  uampere,
  pampere,
  nampere,
  mampere,
  daampere,
  hampere,
  fampere,
  Gampere,
  dampere,
  Pampere,
  Zampere,
  campere,
  aampere,
  Yamp,
  kamp,
  Mamp,
  zamp,
  Eamp,
  yamp,
  Tamp,
  uamp,
  pamp,
  namp,
  mamp,
  daamp,
  hamp,
  famp,
  Gamp,
  damp,
  Pamp,
  Zamp,
  camp,
  aamp,
  Ysecond,
  ksecond,
  Msecond,
  zsecond,
  Esecond,
  ysecond,
  Tsecond,
  usecond,
  psecond,
  nsecond,
  msecond,
  dasecond,
  hsecond,
  fsecond,
  Gsecond,
  dsecond,
  Psecond,
  Zsecond,
  csecond,
  asecond,
  Ymeter,
  kmeter,
  Mmeter,
  zmeter,
  Emeter,
  ymeter,
  Tmeter,
  umeter,
  pmeter,
  nmeter,
  mmeter,
  dameter,
  hmeter,
  fmeter,
  Gmeter,
  dmeter,
  Pmeter,
  Zmeter,
  cmeter,
  ameter,
  Ymetre,
  kmetre,
  Mmetre,
  zmetre,
  Emetre,
  ymetre,
  Tmetre,
  umetre,
  pmetre,
  nmetre,
  mmetre,
  dametre,
  hmetre,
  fmetre,
  Gmetre,
  dmetre,
  Pmetre,
  Zmetre,
  cmetre,
  ametre,
  katal,
  sievert,
  gray,
  becquerel,
  lux,
  lumen,
  henry,
  tesla,
  weber,
  siemens,
  ohm,
  farad,
  volt,
  coulomb,
  watt,
  joule,
  pascal,
  newton,
  hertz,
  steradian,
  radian,
  molar,
  gramme,
  gram,
  kilogramme,
  candle,
  mol,
  mole,
  kelvin,
  ampere,
  amp,
  second,
  kilogram,
  meter,
  metre,
]

del base_units, scaled_units, powered_units, additional_units
