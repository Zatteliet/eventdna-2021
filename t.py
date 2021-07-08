s = "Erdogan wint verkiezingen : Turkije heeft democratische test doorstaan en is voorbeeld voor hele wereld Oppositie klaagt manipulatie aan Turkije heeft een nieuwe democratische test doorstaan en stelt daarmee een voorbeeld voor de hele wereld . Dat heeft Recep Tayyip Erdogan gezegd nadat de kiescommissie bekendmaakte dat het zittende staatshoofd de presidentsverkiezingen al in de eerste ronde had gewonnen . Onze democratie heeft gewonnen , de wil van het volk heeft gewonnen , Turkije heeft gewonnen , zo sprak hij zijn aanhangers toe aan het hoofdkwartier van zijn partij AKP in Ankara . De oppositie , de centrumlinkse Republikeinse Volkspartij ( CHP ) , noemt de uitslag van de verkiezingen manipulatie ."

starts = [93, 60, 36, 19, 15, 0]

for i, token in enumerate(s.split()):
    if int(i) in starts:
        print("-"*10)
    print(i, token)