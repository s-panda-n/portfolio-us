"""
data.screener — Broad stock screener across 280+ tickers.

Scores each ticker on Sharpe ratio, 3-month price momentum, and sector momentum.
Returns all rows ranked by composite score for the discovery panel.
Signal: BULL / BEAR / NEUTRAL (momentum-based)
Action: BUY / SELL / HOLD (composite-score-based)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

# ── Screener universe: {ticker: (sector, description)} ──────────────────────

SCREENER_UNIVERSE: dict[str, tuple[str, str]] = {
    # ── Technology ───────────────────────────────────────────────────────────
    "AAPL":  ("Technology",    "Apple — iPhone, Mac, Services ecosystem"),
    "MSFT":  ("Technology",    "Microsoft — Azure cloud + Copilot AI"),
    "NVDA":  ("Technology",    "Nvidia — dominant AI GPU maker"),
    "GOOGL": ("Technology",    "Alphabet — Google search + cloud + AI"),
    "AMZN":  ("Technology",    "Amazon — AWS cloud + e-commerce"),
    "META":  ("Technology",    "Meta — Facebook/Instagram + AI infra"),
    "AMD":   ("Technology",    "AMD — CPUs and AI accelerator chips"),
    "INTC":  ("Technology",    "Intel — x86 CPUs, foundry transition"),
    "QCOM":  ("Technology",    "Qualcomm — mobile chips + IoT"),
    "AVGO":  ("Technology",    "Broadcom — networking chips + VMware"),
    "TXN":   ("Technology",    "Texas Instruments — analog semiconductors"),
    "MU":    ("Technology",    "Micron — DRAM and NAND memory"),
    "AMAT":  ("Technology",    "Applied Materials — chip manufacturing equipment"),
    "LRCX":  ("Technology",    "Lam Research — wafer fab equipment"),
    "KLAC":  ("Technology",    "KLA Corp — process control for chipmakers"),
    "ASML":  ("Technology",    "ASML — EUV lithography monopoly (Netherlands)"),
    "TSM":   ("Technology",    "TSMC — world's largest chip foundry (Taiwan ADR)"),
    "ORCL":  ("Technology",    "Oracle — cloud ERP + database"),
    "CRM":   ("Technology",    "Salesforce — CRM SaaS leader"),
    "ADBE":  ("Technology",    "Adobe — Creative Cloud + Document Cloud"),
    "NOW":   ("Technology",    "ServiceNow — enterprise workflow automation"),
    "INTU":  ("Technology",    "Intuit — TurboTax, QuickBooks, Credit Karma"),
    "SHOP":  ("Technology",    "Shopify — e-commerce platform"),
    "PYPL":  ("Technology",    "PayPal — digital payments, Venmo"),
    "DELL":  ("Technology",    "Dell — PCs, servers, AI infrastructure"),
    "HPQ":   ("Technology",    "HP Inc — PCs and printers"),
    "HPE":   ("Technology",    "Hewlett Packard Enterprise — cloud infrastructure"),
    "IBM":   ("Technology",    "IBM — hybrid cloud + AI consulting"),
    "CSCO":  ("Technology",    "Cisco — networking infrastructure + security"),
    "TER":   ("Technology",    "Teradyne — semiconductor test equipment"),
    # ── AI / Data ─────────────────────────────────────────────────────────────
    "PLTR":  ("AI/Data",       "Palantir — AI analytics platform for gov & enterprise"),
    "SNOW":  ("AI/Data",       "Snowflake — cloud data warehouse"),
    "AI":    ("AI/Data",       "C3.ai — enterprise AI software"),
    "PATH":  ("AI/Data",       "UiPath — robotic process automation"),
    "DDOG":  ("AI/Data",       "Datadog — cloud observability platform"),
    "MDB":   ("AI/Data",       "MongoDB — NoSQL database cloud"),
    "CFLT":  ("AI/Data",       "Confluent — Apache Kafka streaming platform"),
    "NET":   ("AI/Data",       "Cloudflare — network edge + AI workers"),
    "SMCI":  ("AI/Data",       "Super Micro Computer — AI server rack systems"),
    "ARM":   ("AI/Data",       "ARM Holdings — chip architecture IP"),
    "BBAI":  ("AI/Data",       "BigBear.ai — AI analytics for defense"),
    "MSTR":  ("AI/Data",       "MicroStrategy — enterprise analytics + Bitcoin"),
    # ── Cybersecurity ─────────────────────────────────────────────────────────
    "CRWD":  ("Cybersecurity", "CrowdStrike — AI-native endpoint security leader"),
    "PANW":  ("Cybersecurity", "Palo Alto Networks — next-gen firewall + SASE"),
    "ZS":    ("Cybersecurity", "Zscaler — zero-trust cloud security"),
    "FTNT":  ("Cybersecurity", "Fortinet — network security appliances"),
    "S":     ("Cybersecurity", "SentinelOne — autonomous endpoint protection"),
    "CHKP":  ("Cybersecurity", "Check Point Software — firewall veteran"),
    "CYBR":  ("Cybersecurity", "CyberArk — privileged access management"),
    "VRNS":  ("Cybersecurity", "Varonis — data security and access governance"),
    "SAIC":  ("Cybersecurity", "SAIC — IT services for US government"),
    "CACI":  ("Cybersecurity", "CACI International — defense IT and cyber"),
    # ── Healthcare / Biotech ──────────────────────────────────────────────────
    "JNJ":   ("Healthcare",    "Johnson & Johnson — pharma + MedTech spinoff"),
    "UNH":   ("Healthcare",    "UnitedHealth — largest US health insurer"),
    "PFE":   ("Healthcare",    "Pfizer — vaccines, oncology pipeline"),
    "ABBV":  ("Healthcare",    "AbbVie — Humira + cancer drugs"),
    "MRK":   ("Healthcare",    "Merck — Keytruda cancer blockbuster"),
    "LLY":   ("Healthcare",    "Eli Lilly — Ozempic/Mounjaro weight-loss drugs"),
    "BMY":   ("Healthcare",    "Bristol-Myers Squibb — oncology + immunology"),
    "AMGN":  ("Healthcare",    "Amgen — biotech pioneer, Repatha, Otezla"),
    "GILD":  ("Healthcare",    "Gilead — antivirals, HIV, oncology"),
    "REGN":  ("Healthcare",    "Regeneron — Dupixent, Eylea, cancer drugs"),
    "VRTX":  ("Healthcare",    "Vertex — cystic fibrosis monopoly, pain drugs"),
    "MRNA":  ("Healthcare",    "Moderna — mRNA vaccines, cancer, respiratory"),
    "BIIB":  ("Healthcare",    "Biogen — neurology, Alzheimer's drug Leqembi"),
    "ISRG":  ("Healthcare",    "Intuitive Surgical — Da Vinci robotic surgery"),
    "MDT":   ("Healthcare",    "Medtronic — cardiac devices, surgical robotics"),
    "SYK":   ("Healthcare",    "Stryker — orthopedic implants, medical devices"),
    "BSX":   ("Healthcare",    "Boston Scientific — cardiology + urology devices"),
    "ABT":   ("Healthcare",    "Abbott Labs — diagnostics, heart devices, nutrition"),
    "TMO":   ("Healthcare",    "Thermo Fisher — lab instruments + CDMO"),
    "DHR":   ("Healthcare",    "Danaher — life science tools + diagnostics"),
    "ZBH":   ("Healthcare",    "Zimmer Biomet — joint replacements"),
    "HCA":   ("Healthcare",    "HCA Healthcare — largest US hospital operator"),
    "CVS":   ("Healthcare",    "CVS Health — pharmacy + health insurance"),
    "CI":    ("Healthcare",    "Cigna — health insurance + pharmacy benefits"),
    "NVO":   ("Healthcare",    "Novo Nordisk — Ozempic, Wegovy weight-loss (ADR)"),
    "AZN":   ("Healthcare",    "AstraZeneca — oncology, respiratory, vaccines (ADR)"),
    # ── Financials ───────────────────────────────────────────────────────────
    "JPM":   ("Financials",    "JPMorgan Chase — largest US bank"),
    "BAC":   ("Financials",    "Bank of America — consumer banking giant"),
    "WFC":   ("Financials",    "Wells Fargo — US retail banking"),
    "GS":    ("Financials",    "Goldman Sachs — investment banking + trading"),
    "MS":    ("Financials",    "Morgan Stanley — wealth management + banking"),
    "C":     ("Financials",    "Citigroup — global consumer + institutional banking"),
    "BRK-B": ("Financials",    "Berkshire Hathaway — Buffett conglomerate"),
    "AXP":   ("Financials",    "American Express — premium credit cards"),
    "V":     ("Financials",    "Visa — global payment network"),
    "MA":    ("Financials",    "Mastercard — global payment network"),
    "COF":   ("Financials",    "Capital One — credit cards + auto loans"),
    "SCHW":  ("Financials",    "Charles Schwab — retail brokerage + banking"),
    "BLK":   ("Financials",    "BlackRock — world's largest asset manager"),
    "SPGI":  ("Financials",    "S&P Global — credit ratings + market intelligence"),
    "MCO":   ("Financials",    "Moody's — credit ratings + analytics"),
    "ICE":   ("Financials",    "Intercontinental Exchange — NYSE operator"),
    "CME":   ("Financials",    "CME Group — futures exchanges (derivatives)"),
    "CB":    ("Financials",    "Chubb — global property & casualty insurance"),
    "PGR":   ("Financials",    "Progressive — auto + home insurance"),
    "AFL":   ("Financials",    "Aflac — supplemental health + life insurance"),
    "MET":   ("Financials",    "MetLife — life + health insurance"),
    # ── Consumer Discretionary ───────────────────────────────────────────────
    "TSLA":  ("Consumer Disc", "Tesla — EVs, energy storage, autonomy"),
    "HD":    ("Consumer Disc", "Home Depot — home improvement retail"),
    "LOW":   ("Consumer Disc", "Lowe's — home improvement #2"),
    "BKNG":  ("Consumer Disc", "Booking Holdings — Booking.com, Priceline"),
    "MAR":   ("Consumer Disc", "Marriott — world's largest hotel chain"),
    "HLT":   ("Consumer Disc", "Hilton — global hotel and resort brand"),
    "MCD":   ("Consumer Disc", "McDonald's — global fast food franchise"),
    "SBUX":  ("Consumer Disc", "Starbucks — premium coffee chain"),
    "NKE":   ("Consumer Disc", "Nike — global athletic footwear and apparel"),
    "LULU":  ("Consumer Disc", "Lululemon — premium athletic apparel"),
    "RH":    ("Consumer Disc", "RH (Restoration Hardware) — luxury home furnishings"),
    "ETSY":  ("Consumer Disc", "Etsy — online marketplace for handmade goods"),
    "DPZ":   ("Consumer Disc", "Domino's Pizza — global delivery franchise"),
    "YUM":   ("Consumer Disc", "Yum! Brands — KFC, Pizza Hut, Taco Bell"),
    "CMG":   ("Consumer Disc", "Chipotle — fast-casual Mexican food chain"),
    "DKNG":  ("Consumer Disc", "DraftKings — online sports betting + fantasy"),
    "F":     ("Consumer Disc", "Ford — F-150 Lightning, legacy auto transition"),
    "GM":    ("Consumer Disc", "General Motors — Ultium EV platform"),
    # ── Consumer Staples ─────────────────────────────────────────────────────
    "PG":    ("Consumer Stap", "Procter & Gamble — Tide, Pampers, Gillette"),
    "KO":    ("Consumer Stap", "Coca-Cola — global beverage giant"),
    "PEP":   ("Consumer Stap", "PepsiCo — beverages + Frito-Lay snacks"),
    "WMT":   ("Consumer Stap", "Walmart — largest US retailer, growing e-commerce"),
    "COST":  ("Consumer Stap", "Costco — membership warehouse, loyal base"),
    "TGT":   ("Consumer Stap", "Target — mass merchandise retail"),
    "CL":    ("Consumer Stap", "Colgate-Palmolive — toothpaste, consumer products"),
    "MDLZ":  ("Consumer Stap", "Mondelez — Oreo, Cadbury, global snacks"),
    "GIS":   ("Consumer Stap", "General Mills — Cheerios, Häagen-Dazs"),
    "KHC":   ("Consumer Stap", "Kraft Heinz — packaged food turnaround"),
    "MO":    ("Consumer Stap", "Altria — Marlboro cigarettes, dividend play"),
    "PM":    ("Consumer Stap", "Philip Morris — international tobacco + IQOS"),
    "STZ":   ("Consumer Stap", "Constellation Brands — Corona, Modelo beer"),
    "HSY":   ("Consumer Stap", "Hershey — chocolate and candy"),
    "CAG":   ("Consumer Stap", "Conagra Brands — packaged foods"),
    # ── Energy (Traditional) ─────────────────────────────────────────────────
    "XOM":   ("Energy",        "ExxonMobil — oil & gas supermajor"),
    "CVX":   ("Energy",        "Chevron — oil & gas, global operations"),
    "COP":   ("Energy",        "ConocoPhillips — pure-play upstream oil"),
    "SLB":   ("Energy",        "SLB (Schlumberger) — oilfield services"),
    "OXY":   ("Energy",        "Occidental Petroleum — Buffett-backed energy"),
    "EOG":   ("Energy",        "EOG Resources — US shale oil"),
    "DVN":   ("Energy",        "Devon Energy — oil & gas exploration"),
    "HAL":   ("Energy",        "Halliburton — oilfield services #2"),
    "MPC":   ("Energy",        "Marathon Petroleum — refining and midstream"),
    "PSX":   ("Energy",        "Phillips 66 — refining + chemicals"),
    "VLO":   ("Energy",        "Valero Energy — largest US refiner"),
    "HES":   ("Energy",        "Hess — oil exploration, Guyana deepwater"),
    "BKR":   ("Energy",        "Baker Hughes — oilfield services + LNG"),
    "FANG":  ("Energy",        "Diamondback Energy — Permian Basin operator"),
    # ── Energy (Clean / Renewables) ───────────────────────────────────────────
    "NEE":   ("Clean Energy",  "NextEra Energy — largest US wind + solar + nuclear"),
    "ENPH":  ("Clean Energy",  "Enphase Energy — microinverters for solar"),
    "SEDG":  ("Clean Energy",  "SolarEdge — solar power optimizers"),
    "FSLR":  ("Clean Energy",  "First Solar — thin-film solar panels"),
    "RUN":   ("Clean Energy",  "Sunrun — residential solar + storage"),
    "PLUG":  ("Clean Energy",  "Plug Power — hydrogen fuel cells"),
    "BE":    ("Clean Energy",  "Bloom Energy — solid oxide fuel cells"),
    "CEG":   ("Clean Energy",  "Constellation Energy — nuclear power + clean energy"),
    "AES":   ("Clean Energy",  "AES Corp — global renewable energy utility"),
    "NOVA":  ("Clean Energy",  "Sunnova Energy — residential solar + storage"),
    # ── Industrials ───────────────────────────────────────────────────────────
    "GE":    ("Industrials",   "GE Aerospace — jet engines, aviation systems"),
    "HON":   ("Industrials",   "Honeywell — aerospace + building automation"),
    "CAT":   ("Industrials",   "Caterpillar — construction and mining equipment"),
    "DE":    ("Industrials",   "Deere — farm equipment + autonomy tech"),
    "MMM":   ("Industrials",   "3M — industrial + consumer + healthcare"),
    "UPS":   ("Industrials",   "UPS — global package delivery"),
    "FDX":   ("Industrials",   "FedEx — express shipping + freight"),
    "LIN":   ("Industrials",   "Linde — industrial gases (world's largest)"),
    "APD":   ("Industrials",   "Air Products — hydrogen + industrial gases"),
    "EMR":   ("Industrials",   "Emerson Electric — automation + HVAC"),
    "ROK":   ("Industrials",   "Rockwell Automation — factory automation"),
    "PH":    ("Industrials",   "Parker Hannifin — motion and control systems"),
    "ETN":   ("Industrials",   "Eaton — power management + electrification"),
    "IR":    ("Industrials",   "Ingersoll Rand — industrial machinery"),
    "CARR":  ("Industrials",   "Carrier Global — HVAC and refrigeration"),
    # ── Defense / Aerospace ───────────────────────────────────────────────────
    "LMT":   ("Defense",       "Lockheed Martin — F-35 jets, hypersonics, space"),
    "RTX":   ("Defense",       "RTX (Raytheon) — missiles, jet engines"),
    "NOC":   ("Defense",       "Northrop Grumman — stealth bombers, space systems"),
    "GD":    ("Defense",       "General Dynamics — tanks, submarines, Gulfstream"),
    "BA":    ("Defense",       "Boeing — commercial jets + defense contracts"),
    "TDY":   ("Defense",       "Teledyne — sensors, imaging, defense electronics"),
    "KTOS":  ("Defense",       "Kratos Defense — drones and unmanned systems"),
    "HWM":   ("Defense",       "Howmet Aerospace — jet engine components"),
    "HII":   ("Defense",       "Huntington Ingalls — US Navy shipbuilder"),
    "AXON":  ("Defense",       "Axon Enterprise — Tasers, body cams, law enforcement AI"),
    # ── Auto / EV ─────────────────────────────────────────────────────────────
    "RIVN":  ("Auto/EV",       "Rivian — EV trucks and Amazon delivery vans"),
    "LCID":  ("Auto/EV",       "Lucid Motors — luxury EVs"),
    "NIO":   ("Auto/EV",       "NIO — Chinese premium EVs"),
    "XPEV":  ("Auto/EV",       "XPeng — Chinese EV with smart driving"),
    "LI":    ("Auto/EV",       "Li Auto — Chinese extended-range EVs"),
    "TM":    ("Auto/EV",       "Toyota — hybrid leader + hydrogen bet"),
    "HMC":   ("Auto/EV",       "Honda — autos + motorcycles + power products"),
    "NKLA":  ("Auto/EV",       "Nikola — hydrogen and battery electric trucks"),
    # ── Materials ────────────────────────────────────────────────────────────
    "DD":    ("Materials",     "DuPont — specialty chemicals + electronics materials"),
    "DOW":   ("Materials",     "Dow Inc — commodity and specialty chemicals"),
    "PPG":   ("Materials",     "PPG Industries — coatings and paints"),
    "SHW":   ("Materials",     "Sherwin-Williams — paints and coatings"),
    "NEM":   ("Materials",     "Newmont — world's largest gold miner"),
    "FCX":   ("Materials",     "Freeport-McMoRan — copper and gold mining"),
    "ALB":   ("Materials",     "Albemarle — lithium for EV batteries"),
    "CF":    ("Materials",     "CF Industries — nitrogen fertilizers"),
    "MOS":   ("Materials",     "Mosaic — potash and phosphate fertilizers"),
    "VMC":   ("Materials",     "Vulcan Materials — aggregates for construction"),
    "MLM":   ("Materials",     "Martin Marietta — aggregates and cement"),
    # ── Real Estate / REITs ───────────────────────────────────────────────────
    "AMT":   ("Real Estate",   "American Tower — wireless tower REIT"),
    "PLD":   ("Real Estate",   "Prologis — industrial logistics REIT"),
    "EQIX":  ("Real Estate",   "Equinix — data center REIT"),
    "SPG":   ("Real Estate",   "Simon Property Group — mall REIT"),
    "O":     ("Real Estate",   "Realty Income — net lease REIT, monthly dividends"),
    "PSA":   ("Real Estate",   "Public Storage — self-storage REIT"),
    "EXR":   ("Real Estate",   "Extra Space Storage — self-storage REIT"),
    "DLR":   ("Real Estate",   "Digital Realty — data center REIT"),
    "CCI":   ("Real Estate",   "Crown Castle — cell tower REIT"),
    "IRM":   ("Real Estate",   "Iron Mountain — document storage + data centers"),
    "VICI":  ("Real Estate",   "VICI Properties — gaming and entertainment REIT"),
    "WPC":   ("Real Estate",   "W. P. Carey — diversified net lease REIT"),
    "WELL":  ("Real Estate",   "Welltower — senior housing and healthcare REIT"),
    # ── Utilities ────────────────────────────────────────────────────────────
    "DUK":   ("Utilities",     "Duke Energy — Southeast US utility"),
    "SO":    ("Utilities",     "Southern Company — regulated utility + Vogtle nuclear"),
    "D":     ("Utilities",     "Dominion Energy — Virginia + South Carolina utility"),
    "EXC":   ("Utilities",     "Exelon — largest US nuclear operator"),
    "ES":    ("Utilities",     "Eversource Energy — New England utility"),
    "AEP":   ("Utilities",     "American Electric Power — US transmission giant"),
    "XEL":   ("Utilities",     "Xcel Energy — Midwest utility + renewables"),
    "PCG":   ("Utilities",     "PG&E — California utility, wildfire risk"),
    "SRE":   ("Utilities",     "Sempra — LNG export + California utility"),
    "PPL":   ("Utilities",     "PPL Corp — UK and US utility"),
    "AWK":   ("Utilities",     "American Water Works — largest US water utility"),
    # ── Communication Services ───────────────────────────────────────────────
    "NFLX":  ("Comm Services", "Netflix — streaming with 300M+ subscribers"),
    "DIS":   ("Comm Services", "Disney — parks, streaming, IP franchise"),
    "CMCSA": ("Comm Services", "Comcast — cable + NBCUniversal + Peacock"),
    "T":     ("Comm Services", "AT&T — telecom + fiber, post-media spinoff"),
    "VZ":    ("Comm Services", "Verizon — US wireless and broadband"),
    "TMUS":  ("Comm Services", "T-Mobile — fastest-growing US wireless"),
    "SPOT":  ("Comm Services", "Spotify — audio streaming + podcasts"),
    "ROKU":  ("Comm Services", "Roku — streaming OS and ad platform"),
    "WBD":   ("Comm Services", "Warner Bros. Discovery — Max streaming + CNN"),
    "PARA":  ("Comm Services", "Paramount — CBS + Paramount+ streaming"),
    "SNAP":  ("Comm Services", "Snap — Snapchat, AR glasses"),
    "PINS":  ("Comm Services", "Pinterest — visual discovery + shopping ads"),
    # ── Retail / E-commerce ───────────────────────────────────────────────────
    "TJX":   ("Retail",        "TJ Maxx / Marshalls — off-price retail"),
    "EBAY":  ("Retail",        "eBay — online marketplace, C2C and B2C"),
    "BABA":  ("Retail",        "Alibaba — Chinese e-commerce + cloud (ADR)"),
    "JD":    ("Retail",        "JD.com — Chinese direct e-commerce (ADR)"),
    "ULTA":  ("Retail",        "Ulta Beauty — beauty specialty retail"),
    "BBY":   ("Retail",        "Best Buy — consumer electronics retail"),
    "ANF":   ("Retail",        "Abercrombie & Fitch — teen fashion turnaround"),
    "W":     ("Retail",        "Wayfair — online home furnishings"),
    "FIVE":  ("Retail",        "Five Below — discount retail for teens"),
    "DLTR":  ("Retail",        "Dollar Tree — discount retail + Family Dollar"),
    # ── Crypto-adjacent ───────────────────────────────────────────────────────
    "COIN":  ("Crypto",        "Coinbase — largest US crypto exchange"),
    "MARA":  ("Crypto",        "Marathon Digital — Bitcoin mining"),
    "RIOT":  ("Crypto",        "Riot Platforms — Bitcoin mining"),
    "HUT":   ("Crypto",        "Hut 8 — Bitcoin mining and AI data centers"),
    "CLSK":  ("Crypto",        "CleanSpark — Bitcoin mining with clean energy"),
    # ── ETFs — broad market ───────────────────────────────────────────────────
    "SPY":   ("ETF",           "S&P 500 ETF — whole US market"),
    "QQQ":   ("ETF",           "Nasdaq-100 ETF — top 100 US non-financial"),
    "IWM":   ("ETF",           "Russell 2000 — US small-cap"),
    "VTI":   ("ETF",           "Vanguard Total Market — entire US equity market"),
    "DIA":   ("ETF",           "Dow Jones ETF — 30 blue-chip stocks"),
    # ── ETFs — international ─────────────────────────────────────────────────
    "EFA":   ("ETF",           "Developed markets ETF (Europe, Japan, Australia)"),
    "EEM":   ("ETF",           "Emerging markets ETF (China, India, Brazil)"),
    "VWO":   ("ETF",           "Vanguard EM — broad emerging markets"),
    "INDA":  ("ETF",           "iShares India — India equity market"),
    "EWJ":   ("ETF",           "iShares Japan — Japanese equity market"),
    # ── ETFs — sector SPDR ───────────────────────────────────────────────────
    "XLK":   ("ETF",           "Technology Select SPDR"),
    "XLE":   ("ETF",           "Energy Select SPDR"),
    "XLF":   ("ETF",           "Financial Select SPDR"),
    "XLV":   ("ETF",           "Health Care Select SPDR"),
    "XLI":   ("ETF",           "Industrial Select SPDR"),
    "XLY":   ("ETF",           "Consumer Discretionary SPDR"),
    "XLP":   ("ETF",           "Consumer Staples SPDR"),
    "XLRE":  ("ETF",           "Real Estate Select SPDR"),
    "XLU":   ("ETF",           "Utilities Select SPDR"),
    "XLB":   ("ETF",           "Materials Select SPDR"),
    "XLC":   ("ETF",           "Communication Services SPDR"),
    # ── ETFs — alternatives ───────────────────────────────────────────────────
    "GLD":   ("ETF",           "Gold ETF — inflation hedge"),
    "SLV":   ("ETF",           "Silver ETF — industrial + precious metal"),
    "GSG":   ("ETF",           "Commodities basket ETF"),
    "VNQ":   ("ETF",           "US REIT ETF — commercial real estate"),
    "DBA":   ("ETF",           "Agriculture commodities ETF"),
    "USO":   ("ETF",           "US Oil Fund — crude oil prices"),
    "JETS":  ("ETF",           "US Global Jets ETF — airline stocks"),
    "TLT":   ("ETF",           "iShares 20+ Year Treasury Bond"),
    "AGG":   ("ETF",           "iShares Core US Aggregate Bond"),
    "HYG":   ("ETF",           "iShares High Yield Corporate Bond"),
}

# Maps screener tickers to their SPDR sector ETF for momentum scoring
_TICKER_TO_SECTOR_ETF: dict[str, str] = {
    # Technology
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "GOOGL": "XLK",
    "AMZN": "XLK", "META": "XLK", "AMD": "XLK", "INTC": "XLK",
    "QCOM": "XLK", "AVGO": "XLK", "TXN": "XLK", "MU": "XLK",
    "AMAT": "XLK", "LRCX": "XLK", "KLAC": "XLK", "ASML": "XLK",
    "TSM": "XLK", "ORCL": "XLK", "CRM": "XLK", "ADBE": "XLK",
    "NOW": "XLK", "INTU": "XLK", "SHOP": "XLK", "PYPL": "XLK",
    "DELL": "XLK", "HPQ": "XLK", "HPE": "XLK", "IBM": "XLK",
    "CSCO": "XLK", "TER": "XLK",
    # AI/Data (mapped to XLK)
    "PLTR": "XLK", "SNOW": "XLK", "AI": "XLK", "PATH": "XLK",
    "DDOG": "XLK", "MDB": "XLK", "CFLT": "XLK", "NET": "XLK",
    "SMCI": "XLK", "ARM": "XLK", "BBAI": "XLK", "MSTR": "XLK",
    # Cybersecurity (mapped to XLK)
    "CRWD": "XLK", "PANW": "XLK", "ZS": "XLK", "FTNT": "XLK",
    "S": "XLK", "CHKP": "XLK", "CYBR": "XLK", "VRNS": "XLK",
    "SAIC": "XLI", "CACI": "XLI",
    # Healthcare
    "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV", "ABBV": "XLV",
    "MRK": "XLV", "LLY": "XLV", "BMY": "XLV", "AMGN": "XLV",
    "GILD": "XLV", "REGN": "XLV", "VRTX": "XLV", "MRNA": "XLV",
    "BIIB": "XLV", "ISRG": "XLV", "MDT": "XLV", "SYK": "XLV",
    "BSX": "XLV", "ABT": "XLV", "TMO": "XLV", "DHR": "XLV",
    "ZBH": "XLV", "HCA": "XLV", "CVS": "XLV", "CI": "XLV",
    "NVO": "XLV", "AZN": "XLV",
    # Financials
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF",
    "MS": "XLF", "C": "XLF", "BRK-B": "XLF", "AXP": "XLF",
    "V": "XLF", "MA": "XLF", "COF": "XLF", "SCHW": "XLF",
    "BLK": "XLF", "SPGI": "XLF", "MCO": "XLF", "ICE": "XLF",
    "CME": "XLF", "CB": "XLF", "PGR": "XLF", "AFL": "XLF",
    "MET": "XLF",
    # Consumer Discretionary
    "TSLA": "XLY", "HD": "XLY", "LOW": "XLY", "BKNG": "XLY",
    "MAR": "XLY", "HLT": "XLY", "MCD": "XLY", "SBUX": "XLY",
    "NKE": "XLY", "LULU": "XLY", "RH": "XLY", "ETSY": "XLY",
    "DPZ": "XLY", "YUM": "XLY", "CMG": "XLY", "DKNG": "XLY",
    "F": "XLY", "GM": "XLY",
    # Consumer Staples
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP",
    "COST": "XLP", "TGT": "XLP", "CL": "XLP", "MDLZ": "XLP",
    "GIS": "XLP", "KHC": "XLP", "MO": "XLP", "PM": "XLP",
    "STZ": "XLP", "HSY": "XLP", "CAG": "XLP",
    # Energy
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE",
    "OXY": "XLE", "EOG": "XLE", "DVN": "XLE", "HAL": "XLE",
    "MPC": "XLE", "PSX": "XLE", "VLO": "XLE", "HES": "XLE",
    "BKR": "XLE", "FANG": "XLE",
    # Clean Energy (mapped to XLU)
    "NEE": "XLU", "ENPH": "XLK", "SEDG": "XLK", "FSLR": "XLK",
    "RUN": "XLU", "PLUG": "XLI", "BE": "XLI", "CEG": "XLU",
    "AES": "XLU", "NOVA": "XLU",
    # Industrials
    "GE": "XLI", "HON": "XLI", "CAT": "XLI", "DE": "XLI",
    "MMM": "XLI", "UPS": "XLI", "FDX": "XLI", "LIN": "XLB",
    "APD": "XLB", "EMR": "XLI", "ROK": "XLI", "PH": "XLI",
    "ETN": "XLI", "IR": "XLI", "CARR": "XLI",
    # Defense
    "LMT": "XLI", "RTX": "XLI", "NOC": "XLI", "GD": "XLI",
    "BA": "XLI", "TDY": "XLI", "KTOS": "XLI", "HWM": "XLI",
    "HII": "XLI", "AXON": "XLI",
    # Auto/EV
    "RIVN": "XLY", "LCID": "XLY", "NIO": "XLY", "XPEV": "XLY",
    "LI": "XLY", "TM": "XLY", "HMC": "XLY", "NKLA": "XLY",
    # Materials
    "DD": "XLB", "DOW": "XLB", "PPG": "XLB", "SHW": "XLB",
    "NEM": "XLB", "FCX": "XLB", "ALB": "XLB", "CF": "XLB",
    "MOS": "XLB", "VMC": "XLB", "MLM": "XLB",
    # Real Estate
    "AMT": "XLRE", "PLD": "XLRE", "EQIX": "XLRE", "SPG": "XLRE",
    "O": "XLRE", "PSA": "XLRE", "EXR": "XLRE", "DLR": "XLRE",
    "CCI": "XLRE", "IRM": "XLRE", "VICI": "XLRE", "WPC": "XLRE",
    "WELL": "XLRE",
    # Utilities
    "DUK": "XLU", "SO": "XLU", "D": "XLU", "EXC": "XLU",
    "ES": "XLU", "AEP": "XLU", "XEL": "XLU", "PCG": "XLU",
    "SRE": "XLU", "PPL": "XLU", "AWK": "XLU",
    # Communication Services
    "NFLX": "XLC", "DIS": "XLC", "CMCSA": "XLC", "T": "XLC",
    "VZ": "XLC", "TMUS": "XLC", "SPOT": "XLC", "ROKU": "XLC",
    "WBD": "XLC", "PARA": "XLC", "SNAP": "XLC", "PINS": "XLC",
    # Retail
    "TJX": "XLP", "EBAY": "XLY", "BABA": "XLY", "JD": "XLY",
    "ULTA": "XLY", "BBY": "XLY", "ANF": "XLY", "W": "XLY",
    "FIVE": "XLY", "DLTR": "XLP",
    # Crypto (mapped to XLK loosely)
    "COIN": "XLF", "MARA": "XLK", "RIOT": "XLK", "HUT": "XLK",
    "CLSK": "XLK",
}

TRADING_DAYS = 252


def run_screener(
    period: str = "3mo",
    sector_returns: dict[str, float] | None = None,
    sentiment_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Score all SCREENER_UNIVERSE tickers and return all rows ranked by composite score.

    Composite = 0.45 * Sharpe + 0.35 * 3-month momentum + 0.20 * sector momentum
    Signal: BULL / BEAR / NEUTRAL (based on 3-month momentum)
    Action: BUY / SELL / HOLD (based on composite score)

    Returns DataFrame with columns:
        rank, ticker, sector, signal, action, sharpe, return_3m_%,
        sector_1m_%, composite_score, description
    """
    empty = pd.DataFrame(columns=[
        "rank", "ticker", "sector", "signal", "action",
        "sharpe", "return_3m_%", "sector_1m_%", "composite_score", "description",
    ])

    tickers = list(dict.fromkeys(SCREENER_UNIVERSE.keys()))

    try:
        raw = yf.download(
            tickers, period=period, interval="1d",
            progress=False, auto_adjust=True, group_by="ticker",
        )
        if isinstance(raw.columns, pd.MultiIndex):
            level0 = raw.columns.get_level_values(0)
            close = (
                raw.xs("Close", axis=1, level=0)
                if "Close" in level0
                else raw.xs("Close", axis=1, level=1)
            )
        elif "Close" in raw.columns:
            close = raw["Close"].to_frame()
        else:
            close = raw
    except Exception:
        return empty

    rows: list[dict] = []
    for ticker, (sector, description) in SCREENER_UNIVERSE.items():
        if ticker not in close.columns:
            continue
        s = close[ticker].dropna()
        if len(s) < 20:
            continue

        daily = s.pct_change().dropna()
        n = len(daily)
        ann_vol = float(daily.std() * np.sqrt(TRADING_DAYS))
        ann_ret = float((1 + daily).prod() ** (TRADING_DAYS / n) - 1)
        sharpe_val = ann_ret / ann_vol if ann_vol > 1e-6 else 0.0
        momentum = float(s.iloc[-1] / s.iloc[0] - 1)  # actual period return

        etf = _TICKER_TO_SECTOR_ETF.get(ticker)
        sec_ret_pct = sector_returns.get(etf, 0.0) if sector_returns and etf else 0.0

        # Signal: sentiment override first, then momentum
        sent = (sentiment_map or {}).get(ticker, "")
        if sent == "BULLISH":
            signal = "BULL"
        elif sent == "BEARISH":
            signal = "BEAR"
        elif momentum > 0.15:
            signal = "BULL"
        elif momentum < -0.10:
            signal = "BEAR"
        else:
            signal = "NEUTRAL"

        rows.append({
            "ticker":       ticker,
            "sector":       sector,
            "description":  description,
            "sharpe":       round(sharpe_val, 2),
            "return_3m_%":  round(momentum * 100, 1),
            "sector_1m_%":  round(sec_ret_pct, 1),
            "_sh":          sharpe_val,
            "_mom":         momentum,
            "_sec":         sec_ret_pct / 100,
            "signal":       signal,
        })

    if not rows:
        return empty

    df = pd.DataFrame(rows)

    # Min-max normalise to [0, 1]
    for col in ["_sh", "_mom", "_sec"]:
        mn, mx = df[col].min(), df[col].max()
        df[f"{col}_n"] = (df[col] - mn) / (mx - mn + 1e-9)

    df["composite_score"] = (
        0.45 * df["_sh_n"] +
        0.35 * df["_mom_n"] +
        0.20 * df["_sec_n"]
    ).round(3)

    # Action based on composite score
    df["action"] = df["composite_score"].apply(
        lambda v: "BUY" if v >= 0.62 else ("SELL" if v <= 0.35 else "HOLD")
    )

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    return df[[
        "rank", "ticker", "sector", "signal", "action",
        "sharpe", "return_3m_%", "sector_1m_%", "composite_score", "description",
    ]]
