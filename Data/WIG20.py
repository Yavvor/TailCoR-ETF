from datetime import datetime, date


class WIG20:
    def __init__(self):
        """
        Inicjalizuje klasę i ładuje historię składu indeksu.
        Dane są automatycznie sortowane od najnowszych do najstarszych.
        """
        self._raw_data = [
            {
                "DataStart": "2017-12-15",
                "PKO": 15.00,  # PKO BP
                "PKN": 15.00,  # ORLEN (dawniej PKN)
                "PZU": 11.97,  # PZU
                "PEO": 9.06,  # PEKAO
                "KGH": 7.02,  # KGHM
                "SPL": 5.39,  # SANTANDER (dawniej BZWBK)
                "LPP": 4.84,  # LPP
                "PGN": 4.55,  # PGNIG (obecnie część ORL, ale w 2017 osobno)
                "PGE": 4.39,  # PGE
                "CCC": 3.37,  # CCC
                "ALR": 2.88,  # ALIOR
                "MBK": 2.75,  # MBANK
                "LTS": 2.47,  # LOTOS (obecnie część ORL)
                "CPS": 2.44,  # CYFRPLSAT
                "JSW": 2.20,  # JSW
                "OPL": 1.62,  # ORANGEPL (dawniej TPSA)
                "TPE": 1.48,  # TAURONPE
                "ACP": 1.48,  # ASSECOPOL
                "ENG": 1.09,  # ENERGA (obecnie część ORL)
                "EUA": 1.00  # EUROCASH
            },
            {
                "DataStart": "2018-03-16",
                "PKO": 15.00,  # PKO BP
                "PKN": 12.34,  # ORLEN (dawniej PKNORLEN)
                "PZU": 11.48,  # PZU
                "PEO": 9.34,  # PEKAO
                "KGH": 6.80,  # KGHM
                "LPP": 5.57,  # LPP
                "SPL": 5.30,  # SANTANDER (dawniej BZWBK)
                "PGN": 4.64,  # PGNIG (obecnie część Grupy ORLEN)
                "PGE": 4.00,  # PGE
                "CCC": 3.41,  # CCC
                "CDR": 3.38,  # CD PROJEKT
                "ALR": 3.32,  # ALIOR BANK
                "CPS": 2.86,  # CYFROWY POLSAT
                "MBK": 2.80,  # MBANK
                "JSW": 2.38,  # JSW
                "LTS": 2.25,  # LOTOS (obecnie część Grupy ORLEN)
                "OPL": 1.73,  # ORANGE POLSKA
                "TPE": 1.35,  # TAURON PE
                "ENG": 1.13,  # ENERGA (obecnie część Grupy ORLEN)
                "EUA": 0.92  # EUROCASH
            },
            {
                "DataStart": "2018-06-15",
                "PKO": 15.00,  # PKO BP
                "PKN": 11.95,  # ORLEN (dawniej PKNORLEN)
                "PZU": 10.79,  # PZU
                "PEO": 9.06,  # PEKAO
                "KGH": 6.28,  # KGHM
                "LPP": 5.78,  # LPP
                "SPL": 5.56,  # SANTANDER (dawniej BZWBK)
                "PGN": 4.85,  # PGNIG (obecnie część ORLEN)
                "CDR": 4.64,  # CD PROJEKT
                "PGE": 4.03,  # PGE
                "CCC": 3.68,  # CCC
                "CPS": 3.31,  # CYFROWY POLSAT
                "ALR": 3.15,  # ALIOR
                "MBK": 2.74,  # MBANK
                "LTS": 2.42,  # LOTOS (obecnie część ORLEN)
                "JSW": 2.20,  # JSW
                "OPL": 1.60,  # ORANGE POLSKA
                "TPE": 1.09,  # TAURON
                "ENG": 0.99,  # ENERGA (obecnie część ORLEN)
                "EUA": 0.90  # EUROCASH
            },
            {
                "DataStart": "2018-09-21",
                "PKO": 15.00,  # PKO BP
                "PKN": 13.51,  # ORLEN (dawniej PKNORLEN)
                "PZU": 11.73,  # PZU
                "PEO": 8.89,  # PEKAO
                "CDR": 6.60,  # CD PROJEKT
                "KGH": 5.70,  # KGHM
                "SPL": 5.62,  # SANTANDER (dawniej BZWBK)
                "LPP": 5.56,  # LPP
                "PGN": 4.19,  # PGNIG (obecnie część Grupy ORLEN)
                "PGE": 3.29,  # PGE
                "CCC": 3.00,  # CCC
                "CPS": 2.91,  # CYFROWY POLSAT
                "LTS": 2.89,  # LOTOS (obecnie część Grupy ORLEN)
                "ALR": 2.84,  # ALIOR BANK
                "MBK": 2.47,  # MBANK
                "JSW": 1.85,  # JSW
                "OPL": 1.49,  # ORANGE POLSKA
                "TPE": 1.00,  # TAURON PE
                "ENG": 0.80,  # ENERGA (obecnie część Grupy ORLEN)
                "EUA": 0.65  # EUROCASH
            },
            {
                "DataStart": "2018-12-21",
                "PKO": 15.00,  # PKO BP
                "PKN": 14.43,  # ORLEN (dawniej PKNORLEN)
                "PZU": 11.47,  # PZU
                "PEO": 8.16,  # PEKAO
                "KGH": 6.00,  # KGHM
                "SPL": 5.92,  # SANTANDER (w liście jako SANPL / dawniej BZWBK)
                "LPP": 5.09,  # LPP
                "PGN": 4.98,  # PGNIG (obecnie wycofane, część ORL)
                "PGE": 4.65,  # PGE
                "CDR": 4.62,  # CD PROJEKT
                "LTS": 3.24,  # LOTOS (obecnie wycofane, część ORL)
                "CPS": 3.00,  # CYFROWY POLSAT
                "CCC": 2.57,  # CCC
                "MBK": 2.55,  # MBANK
                "ALR": 2.24,  # ALIOR BANK
                "JSW": 1.87,  # JSW
                "OPL": 1.55,  # ORANGE POLSKA
                "TPE": 1.11,  # TAURON PE
                "ENG": 0.91,  # ENERGA (obecnie wycofane, część ORL)
                "EUA": 0.65  # EUROCASH
            },
            {
                "DataStart": "2019-03-15",
                "PKO": 15.00,  # PKO BP
                "PKN": 13.55,  # ORLEN (dawniej PKNORLEN)
                "PZU": 10.99,  # PZU
                "PEO": 8.01,  # PEKAO
                "KGH": 5.90,  # KGHM
                "SPL": 5.75,  # SANTANDER (dawniej SANPL / BZWBK)
                "CDR": 5.57,  # CD PROJEKT
                "PGN": 5.08,  # PGNIG (wycofane z obrotu, obecnie część ORL)
                "LPP": 4.59,  # LPP
                "PGE": 4.35,  # PGE
                "LTS": 3.70,  # LOTOS (wycofane z obrotu, obecnie część ORL)
                "CPS": 3.11,  # CYFROWY POLSAT
                "MBK": 2.59,  # MBANK
                "ALR": 2.32,  # ALIOR BANK
                "CCC": 2.21,  # CCC
                "DNP": 2.03,  # DINO POLSKA (ticker rynkowy: DNP)
                "OPL": 1.60,  # ORANGE POLSKA
                "JSW": 1.50,  # JSW
                "TPE": 1.10,  # TAURON PE
                "PLY": 1.05  # PLAY (wycofane z obrotu w 2021 r.)
            },
            {
                "DataStart": "2019-06-21",
                "PKO": 15.00,  # PKO BP
                "PKN": 12.65,  # ORLEN (dawniej PKNORLEN)
                "PZU": 11.05,  # PZU
                "PEO": 8.07,  # PEKAO
                "CDR": 7.14,  # CD PROJEKT
                "KGH": 6.24,  # KGHM
                "SPL": 6.01,  # SANTANDER (dawniej SANPL / BZWBK)
                "LPP": 4.58,  # LPP
                "PGN": 4.37,  # PGNIG (obecnie część Grupy ORLEN)
                "PGE": 3.50,  # PGE
                "LTS": 3.46,  # LOTOS (obecnie część Grupy ORLEN)
                "CPS": 3.32,  # CYFROWY POLSAT
                "DNP": 2.94,  # DINO POLSKA
                "MBK": 2.46,  # MBANK
                "ALR": 2.20,  # ALIOR BANK
                "CCC": 2.17,  # CCC
                "OPL": 1.53,  # ORANGE POLSKA
                "JSW": 1.34,  # JSW
                "PLY": 1.12,  # PLAY (spółka wycofana z giełdy w 2021 r.)
                "TPE": 0.84  # TAURON PE
            },
            {
                "DataStart": "2019-09-20",
                "PKO": 15.0000,  # PKO BP
                "PKN": 13.3358,  # ORLEN (dawniej PKNORLEN)
                "PZU": 10.9415,  # PZU
                "CDR": 8.3145,  # CD PROJEKT
                "PEO": 8.0104,  # PEKAO
                "KGH": 5.2988,  # KGHM
                "SPL": 5.2320,  # SANTANDER (dawniej SANPL / BZWBK)
                "LPP": 4.8156,  # LPP
                "PGN": 3.9163,  # PGNIG (obecnie wycofana, część ORL)
                "CPS": 3.9034,  # CYFROWY POLSAT
                "DNP": 3.6213,  # DINO POLSKA
                "LTS": 3.5688,  # LOTOS (obecnie wycofana, część ORL)
                "PGE": 3.0607,  # PGE
                "MBK": 2.1711,  # MBANK
                "OPL": 1.9864,  # ORANGE POLSKA
                "ALR": 1.9028,  # ALIOR BANK
                "CCC": 1.8751,  # CCC
                "PLY": 1.3988,  # PLAY (wycofana z obrotu w 2021 r.)
                "JSW": 0.8612,  # JSW
                "TPE": 0.7857  # TAURON PE
            },
            {
                "DataStart": "2019-12-20",
                "PKO": 15.0000,  # PKO BP
                "PKN": 13.2333,  # ORLEN (dawniej PKNORLEN)
                "PZU": 10.4764,  # PZU
                "CDR": 8.7067,  # CD PROJEKT
                "PEO": 8.1158,  # PEKAO
                "KGH": 5.9946,  # KGHM
                "LPP": 5.5380,  # LPP
                "SPL": 4.7578,  # SANTANDER (w liście jako SANPL)
                "CPS": 3.8267,  # CYFROWY POLSAT
                "PGN": 3.6731,  # PGNIG (obecnie część ORL)
                "LTS": 3.6719,  # LOTOS (obecnie część ORL)
                "PGE": 3.3001,  # PGE
                "DNP": 3.1219,  # DINO POLSKA (ticker rynkowy: DNP)
                "MBK": 2.4581,  # MBANK
                "OPL": 2.0860,  # ORANGE POLSKA
                "PLY": 1.7739,  # PLAY (wycofane z obrotu)
                "CCC": 1.5305,  # CCC
                "ALR": 1.3146,  # ALIOR BANK
                "TPE": 0.8871,  # TAURON PE
                "JSW": 0.5336  # JSW
            },
            {
                "DataStart": "2020-03-20",
                "PKO": 14.7598,  # PKO BP
                "PZU": 11.2789,  # PZU
                "CDR": 11.1498,  # CD PROJEKT
                "PKN": 10.5972,  # ORLEN (dawniej PKNORLEN)
                "PEO": 8.8969,  # PEKAO
                "KGH": 6.2005,  # KGHM
                "LPP": 5.3564,  # LPP
                "SPL": 4.9329,  # SANTANDER (w liście jako SANPL / dawniej BZWBK)
                "DNP": 3.9993,  # DINO POLSKA (ticker rynkowy: DNP)
                "CPS": 3.9354,  # CYFROWY POLSAT
                "LTS": 3.2361,  # LOTOS (obecnie część Grupy ORLEN)
                "PGN": 2.9545,  # PGNIG (obecnie część Grupy ORLEN)
                "MBK": 2.4364,  # MBANK
                "OPL": 2.4106,  # ORANGE POLSKA
                "PGE": 2.2564,  # PGE
                "PLY": 1.9903,  # PLAY (wycofany z giełdy w 2021 r.)
                "CCC": 1.2993,  # CCC
                "ALR": 1.1831,  # ALIOR BANK
                "TPE": 0.7047,  # TAURON PE
                "JSW": 0.4215  # JSW (poprawiony udział wg sumy wag)
            },
            {
                "DataStart": "2020-06-19",
                "CDR": 14.9998,  # CD PROJEKT
                "PKO": 11.7935,  # PKO BP
                "PKN": 11.7463,  # ORLEN (dawniej PKNORLEN)
                "PZU": 10.7086,  # PZU
                "KGH": 7.0885,  # KGHM
                "PEO": 5.8098,  # PEKAO
                "DNP": 5.4605,  # DINO POLSKA (ticker rynkowy: DNP)
                "LPP": 5.2818,  # LPP
                "CPS": 4.4106,  # CYFROWY POLSAT
                "PGN": 3.9665,  # PGNIG (obecnie część Grupy ORLEN)
                "LTS": 3.2975,  # LOTOS (obecnie część Grupy ORLEN)
                "SPL": 3.2193,  # SANTANDER (dawniej SANPL / BZWBK)
                "PLY": 2.9137,  # PLAY (wycofany z giełdy w 2021 r.)
                "OPL": 2.6981,  # ORANGE POLSKA
                "PGE": 2.1712,  # PGE
                "MBK": 1.5541,  # MBANK
                "CCC": 0.8791,  # CCC
                "TPE": 0.7790,  # TAURON PE
                "ALR": 0.7763,  # ALIOR BANK
                "JSW": 0.4458  # JSW
            },
            {
                "DataStart": "2020-09-18",
                "CDR": 14.9999,  # CD PROJEKT
                "PKO": 10.8541,  # PKO BP
                "KGH": 10.4564,  # KGHM
                "PZU": 9.1748,  # PZU
                "PKN": 8.7767,  # ORLEN (dawniej PKNORLEN)
                "DNP": 6.5425,  # DINO POLSKA (ticker rynkowy: DNP)
                "LPP": 5.4188,  # LPP
                "PEO": 5.3637,  # PEKAO
                "PGN": 4.9385,  # PGNIG (obecnie część Grupy ORLEN)
                "CPS": 4.5249,  # CYFROWY POLSAT
                "PGE": 3.0755,  # PGE
                "SPL": 2.9629,  # SANTANDER (w liście jako SANPL / dawniej BZWBK)
                "OPL": 2.7953,  # ORANGE POLSKA
                "PLY": 2.6968,  # PLAY (wycofany z giełdy w 2021 r.)
                "LTS": 2.0667,  # LOTOS (obecnie część Grupy ORLEN)
                "TPE": 1.5073,  # TAURON PE
                "MBK": 1.4322,  # MBANK
                "CCC": 1.2384,  # CCC
                "ALR": 0.7127,  # ALIOR BANK
                "JSW": 0.4618  # JSW
            },
            {
                "DataStart": "2020-12-18",
                "CDR": 12.8034,  # CD PROJEKT
                "ALE": 12.3760,  # ALLEGRO
                "PKO": 11.7524,  # PKO BP
                "KGH": 9.9285,  # KGHM
                "PZU": 7.7009,  # PZU
                "PKN": 7.3878,  # ORLEN (dawniej PKNORLEN)
                "DNP": 6.6568,  # DINO POLSKA
                "PEO": 5.1351,  # PEKAO
                "PGN": 3.9747,  # PGNIG (obecnie część Grupy ORLEN)
                "LPP": 3.8774,  # LPP
                "CPS": 3.7432,  # CYFROWY POLSAT
                "SPL": 3.0317,  # SANTANDER (w liście jako SANPL / dawniej BZWBK)
                "PGE": 2.4687,  # PGE
                "OPL": 2.1145,  # ORANGE POLSKA
                "ACP": 1.9690,  # ASSECO POLAND (w liście jako ASSECOPOL)
                "LTS": 1.4662,  # LOTOS (obecnie część Grupy ORLEN)
                "TPE": 1.1492,  # TAURON PE
                "CCC": 1.1334,  # CCC
                "ALR": 0.7719,  # ALIOR BANK
                "JSW": 0.5592  # JSW
            },
            {
                "DataStart": "2021-03-19",
                "KGH": 13.3000,  # KGHM
                "PKO": 12.3549,  # PKO BP
                "ALE": 9.8124,  # ALLEGRO
                "CDR": 8.8439,  # CD PROJEKT
                "PKN": 8.2414,  # ORLEN (dawniej PKNORLEN)
                "PZU": 8.1632,  # PZU
                "DNP": 5.6470,  # DINO POLSKA (ticker rynkowy: DNP)
                "PEO": 5.5677,  # PEKAO
                "PGN": 4.4764,  # PGNIG (obecnie część Grupy ORLEN)
                "LPP": 4.0140,  # LPP
                "CPS": 3.6712,  # CYFROWY POLSAT
                "SPL": 3.3931,  # SANTANDER (w liście jako SANPL / dawniej BZWBK)
                "PGE": 2.4822,  # PGE
                "OPL": 1.8448,  # ORANGE POLSKA
                "CCC": 1.8147,  # CCC
                "ACP": 1.7326,  # ASSECO POLAND (w liście jako ASSECOPOL)
                "LTS": 1.6937,  # LOTOS (obecnie część Grupy ORLEN)
                "TPE": 1.3240,  # TAURON PE
                "JSW": 0.9856,  # JSW
                "MRC": 0.6372  # MERCATOR MEDICAL (ticker rynkowy: MRC)
            },
            {
                "DataStart": "2021-06-18",
                "PKO": 13.6172,  # PKO BP
                "KGH": 11.5909,  # KGHM
                "ALE": 9.5175,  # ALLEGRO
                "PKN": 9.4273,  # ORLEN (dawniej PKNORLEN)
                "PZU": 8.6433,  # PZU
                "PEO": 6.9410,  # PEKAO
                "DNP": 5.6978,  # DINO POLSKA (ticker rynkowy: DNP)
                "CDR": 5.2713,  # CD PROJEKT
                "LPP": 4.6317,  # LPP
                "PGN": 4.5331,  # PGNIG (obecnie część Grupy ORLEN)
                "SPL": 3.6635,  # SANTANDER (w liście jako SANPL / dawniej BZWBK)
                "CPS": 3.3566,  # CYFROWY POLSAT
                "PGE": 3.1287,  # PGE
                "CCC": 1.9501,  # CCC
                "LTS": 1.8911,  # LOTOS (obecnie część Grupy ORLEN)
                "OPL": 1.8613,  # ORANGE POLSKA
                "ACP": 1.7023,  # ASSECO POLAND (w liście jako ASSECOPOL)
                "TPE": 1.4219,  # TAURON PE
                "JSW": 0.7215,  # JSW
                "MRC": 0.4317  # MERCATOR MEDICAL (ticker rynkowy: MRC)
            },
            {
                "DataStart": "2021-09-17",
                "PKO": 13.5860,  # PKO BP
                "ALE": 10.2408,  # ALLEGRO
                "KGH": 9.5939,  # KGHM
                "PZU": 9.2513,  # PZU
                "PKN": 8.4267,  # ORLEN (dawniej PKNORLEN)
                "PEO": 7.3545,  # PEKAO
                "DNP": 6.2679,  # DINO POLSKA (ticker rynkowy: DNP)
                "LPP": 5.5437,  # LPP
                "CDR": 4.7745,  # CD PROJEKT
                "PGN": 4.1712,  # PGNIG (obecnie część Grupy ORLEN)
                "CPS": 3.8258,  # CYFROWY POLSAT
                "SPL": 3.6818,  # SANTANDER (w liście jako SANPL / dawniej BZWBK)
                "PGE": 3.0085,  # PGE
                "OPL": 2.1602,  # ORANGE POLSKA
                "LTS": 2.0087,  # LOTOS (obecnie część Grupy ORLEN)
                "ACP": 1.8370,  # ASSECO POLAND (w liście jako ASSECOPOL)
                "CCC": 1.7685,  # CCC
                "TPE": 1.4739,  # TAURON PE
                "JSW": 0.7350,  # JSW
                "MRC": 0.2902  # MERCATOR MEDICAL (ticker rynkowy: MRC)
            },
            {
                "DataStart": "2021-12-17",
                "PKO": 15.0000,  # PKO BP
                "PKN": 9.1638,  # ORLEN (dawniej PKNORLEN)
                "PEO": 8.6353,  # PEKAO
                "PZU": 8.3716,  # PZU
                "KGH": 8.0002,  # KGHM
                "ALE": 7.1942,  # ALLEGRO
                "DNP": 6.6128,  # DINO POLSKA (ticker rynkowy: DNP)
                "LPP": 5.5989,  # LPP
                "CDR": 5.4760,  # CD PROJEKT
                "SPL": 4.7014,  # SANTANDER (w liście jako SANPL / dawniej BZWBK)
                "PGN": 4.0673,  # PGNIG (obecnie część Grupy ORLEN)
                "CPS": 3.8124,  # CYFROWY POLSAT
                "PGE": 3.0422,  # PGE
                "OPL": 2.1966,  # ORANGE POLSKA
                "ACP": 2.1217,  # ASSECO POLAND (w liście jako ASSECOPOL)
                "LTS": 2.0274,  # LOTOS (obecnie część Grupy ORLEN)
                "CCC": 1.6322,  # CCC
                "TPE": 1.2939,  # TAURON PE
                "JSW": 0.8860,  # JSW
                "MRC": 0.1660  # MERCATOR MEDICAL (ticker rynkowy: MRC)
            },
            {
                "DataStart": "2022-03-18",
                "PKO": 15.0000,  # PKO BP
                "PEO": 9.5173,  # PEKAO
                "KGH": 8.7908,  # KGHM
                "PKN": 8.6364,  # ORLEN (dawniej PKNORLEN)
                "PZU": 8.2580,  # PZU
                "LPP": 6.6002,  # LPP
                "DNP": 5.8486,  # DINO POLSKA (ticker rynkowy: DNP)
                "ALE": 5.3062,  # ALLEGRO
                "CDR": 5.1943,  # CD PROJEKT
                "SPL": 4.8652,  # SANTANDER (dawniej SANPL / BZWBK)
                "PGN": 3.6461,  # PGNIG (obecnie część Grupy ORLEN)
                "CPS": 3.2029,  # CYFROWY POLSAT
                "PGE": 2.5337,  # PGE
                "MBK": 2.4878,  # MBANK
                "OPL": 2.2610,  # ORANGE POLSKA
                "PCO": 2.0630,  # PEPCO (oficjalny ticker: PCO)
                "LTS": 1.9824,  # LOTOS (obecnie część Grupy ORLEN)
                "ACP": 1.8796,  # ASSECO POLAND (ticker rynkowy: ACP)
                "JSW": 0.9867,  # JSW
                "CCC": 0.9398  # CCC
            },
            {
                "DataStart": "2022-06-17",
                "PKO": 12.9979,  # PKO BP
                "PKN": 10.1195,  # ORLEN (dawniej PKNORLEN)
                "KGH": 9.1035,  # KGHM
                "PZU": 8.6134,  # PZU
                "PEO": 8.0259,  # PEKAO
                "DNP": 7.1994,  # DINO POLSKA (ticker rynkowy: DNP)
                "LPP": 5.3264,  # LPP
                "ALE": 5.0933,  # ALLEGRO
                "PGN": 4.9083,  # PGNIG (obecnie część Grupy ORLEN)
                "CDR": 4.1740,  # CD PROJEKT
                "SPL": 4.1178,  # SANTANDER (dawniej SANPL / BZWBK)
                "PGE": 3.8510,  # PGE
                "LTS": 2.8793,  # LOTOS (obecnie część Grupy ORLEN)
                "CPS": 2.7482,  # CYFROWY POLSAT (ticker rynkowy: CPS)
                "PCO": 2.4271,  # PEPCO (oficjalny ticker: PCO)
                "ACP": 2.1581,  # ASSECO POLAND (ticker rynkowy: ACP)
                "OPL": 1.9416,  # ORANGE POLSKA
                "JSW": 1.7518,  # JSW
                "MBK": 1.7022,  # MBANK
                "CCC": 0.8613  # CCC
            },
            {
                "DataStart": "2022-09-16",
                "PKN": 14.1769,  # ORLEN (dawniej PKNORLEN)
                "PKO": 10.9226,  # PKO BP
                "PZU": 8.8538,  # PZU
                "DNP": 8.7293,  # DINO POLSKA (ticker rynkowy: DNP)
                "KGH": 7.1448,  # KGHM
                "PEO": 6.5403,  # PEKAO
                "ALE": 5.8782,  # ALLEGRO
                "PGN": 5.2294,  # PGNIG (obecnie część Grupy ORLEN)
                "LPP": 5.2197,  # LPP
                "SPL": 4.1342,  # SANTANDER (dawniej SANPL / BZWBK)
                "PGE": 3.9676,  # PGE
                "CDR": 3.2314,  # CD PROJEKT
                "KTY": 2.8478,  # GRUPA KĘTY (oficjalny ticker: KTY)
                "CPS": 2.5378,  # CYFROWY POLSAT (ticker rynkowy: CPS)
                "PCO": 2.3827,  # PEPCO (oficjalny ticker: PCO)
                "ACP": 2.2328,  # ASSECO POLAND (ticker rynkowy: ACP)
                "OPL": 2.0877,  # ORANGE POLSKA
                "MBK": 1.6400,  # MBANK
                "JSW": 1.4297,  # JSW
                "CCC": 0.8135  # CCC
            }
        ]

        self._history = self._process_data()

    def _process_data(self):
        """
        Metoda wewnętrzna: Konwertuje daty ze stringów na obiekty datetime
        i sortuje historię malejąco (od najnowszej do najstarszej).
        """
        processed_history = []
        for entry in self._raw_data:
            # Kopiujemy słownik, żeby nie modyfikować oryginału
            data_copy = entry.copy()
            # Wyciągamy datę startu i usuwamy ją ze słownika wag
            date_str = data_copy.pop("DataStart")
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                processed_history.append((date_obj, data_copy))
            except ValueError:
                print(f"Błąd formatu daty dla wpisu: {date_str}")

        # Sortowanie: Najnowsze daty na początku listy
        processed_history.sort(key=lambda x: x[0], reverse=True)
        return processed_history

    def _find_composition(self, query_date):
        """
        Znajduje odpowiedni skład indeksu dla podanej daty.
        Logika: Szuka pierwszego wpisu, którego DataStart <= DataZapytania.
        """
        if isinstance(query_date, str):
            try:
                query_date = datetime.strptime(query_date, "%Y-%m-%d").date()
            except ValueError:
                raise ValueError("Nieprawidłowy format daty. Oczekiwany format: 'YYYY-MM-DD'")
        elif not isinstance(query_date, date):
            raise TypeError("Oczekiwany format daty to str (YYYY-MM-DD) lub obiekt datetime.date")

        for start_date, weights in self._history:
            if start_date <= query_date:
                return weights

        return None  # Jeśli data jest starsza niż najstarszy wpis w bazie

    def get_index_weights(self, date):
        """
        Zwraca pełny słownik z wagami (bez klucza DataStart) na dany dzień.
        """
        weights = self._find_composition(date)
        if weights:

            return weights
        else:
            return {}  # Zwraca pusty słownik, jeśli brak danych

    def get_ticker_weight(self, ticker, date):
        """
        Zwraca wagę konkretnego tickera.
        Zwraca 0.0, jeśli spółki nie ma w indeksie w danym dniu.
        """
        weights = self._find_composition(date)

        if weights is None:
            print(f"Ostrzeżenie: Brak danych historycznych dla daty {date}.")
            return None

        # .get(ticker, 0.0) zwróci 0, jeśli spółki nie ma w spisie
        return weights.get(ticker.upper(), 0.0)

    def _find_entry(self, query_date):
        """
        Metoda pomocnicza: Znajduje krotkę (DataStart, Wagi) dla zadanej daty.
        Obsługuje wejście typu 'date' oraz 'str'.
        """
        # Ujednolicenie typu do datetime.date
        if isinstance(query_date, str):
            try:
                target_date = datetime.strptime(query_date, "%Y-%m-%d").date()
            except ValueError:
                raise ValueError("Nieprawidłowy format daty string. Użyj 'YYYY-MM-DD'.")
        elif isinstance(query_date, date):
            # Jeśli to datetime.datetime, pobierz .date(), jeśli date, zostaw bez zmian
            target_date = query_date if type(query_date) is date else query_date.date()
        else:
            raise TypeError("Data musi być typu 'datetime.date' lub 'str'.")

        # Właściwe szukanie w posortowanej historii
        for start_date, weights in self._history:
            if start_date <= target_date:
                return start_date, weights

        return None, None

    def get_last_update_date(self, query_date) -> date:
        """
        Zwraca datę (typ date), kiedy nastąpiła ostatnia aktualizacja indeksu
        obowiązująca w dniu 'query_date'.
        """
        found_date, _ = self._find_entry(query_date)
        return found_date


if __name__=="__main__":

    wig = WIG20()

    # 1. Sprawdzenie wagi konkretnej spółki (np. CD Projekt) w dacie z Twojego przykładu
    waga_cdr = wig.get_ticker_weight("CDR", "2018-05-20")
    print(f"Waga CDR w dniu 2018-05-20: {waga_cdr}%")
    # Powinno być 3.38 (ponieważ data jest po 2018-03-16, ale przed 2021)

    # 2. Sprawdzenie wagi tej samej spółki w "przyszłości" (dane fikcyjne z 2021)
    waga_cdr_nowa = wig.get_ticker_weight("CDR", "2022-01-01")
    print(f"Waga CDR w dniu 2022-01-01: {waga_cdr_nowa}%")
    # Powinno być 12.00 (z nowszego słownika)

    # 3. Pobranie całego zestawu wag
    wszystkie_wagi = wig.get_index_weights("2018-04-01")
    print(f"\nLiczba spółek w indeksie (2018-04-01): {len(wszystkie_wagi)}")
    print("Przykładowe 3 spółki:", list(wszystkie_wagi.items())[:3])