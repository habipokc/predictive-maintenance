# Bir Kestirimci Bakım Modelinin Hikâyesi

## Elimizde ne vardı

Bu proje, UCI deposundaki AI4I 2020 Predictive Maintenance veri setiyle çalışıyor. On bin satır, on dört kolon, bir freze tezgâhının simülasyonu. Verinin sentetik olduğunu baştan söylemek gerekiyor, çünkü sonuçların nasıl okunması gerektiğini bu belirliyor: arıza modları belli kurallarla üretilmiş, dolayısıyla gerçek bir fabrikadan toplanmış veride bulamayacağın kadar temiz. Buna karşılık dengesizlik, etiket tutarsızlığı ve öğrenilemez gürültü gibi gerçek problemlerin hepsi içinde mevcut. Öğrenmek için doğru, üretim vaadinde bulunmak için yanlış bir veri seti.

Kolonların beşi sensör okuması: ortam sıcaklığı ve proses sıcaklığı Kelvin cinsinden, mil devri dakikada devir olarak, tork Newton-metre, takım aşınması dakika. Bir de ürün kalite sınıfı var, düşükten yükseğe L, M ve H. Hedef değişken makinenin o satırda arızalanıp arızalanmadığı. Yanında beş arıza modu kolonu duruyor: takım aşınması arızası TWF, ısı dağıtım arızası HDF, güç arızası PWF, aşırı zorlanma arızası OSF ve rastgele arıza RNF.

On bin satırın üç yüz otuz dokuzu arızalı. Yüzde 3.39. Her yirmi dokuz parçadan biri. Bu oran dengesiz sayılır ama otomotiv sektöründeki hata oranlarının milyonda mertebesinde olduğunu düşününce fazlasıyla iyimser kalıyor. Bu yüzden projenin başında ikinci bir rejim kurdum: dokuz bin altı yüz altmış bir sağlam satırın tamamını koruyup arızaları elliye indirdim. Bu yüzde 0.515 oranında bir stres varyantı verdi. Alternatifi, toplam satır sayısını on binde tutmak için negatiflerden de atmaktı; onu seçmedim çünkü gereksiz karmaşıklık ve bilgi kaybı yaratırdı, üstelik kurgu da bozulurdu. Sadece pozitifleri seyrelttiğim için elimdeki senaryo "aynı hat, aynı üretim hacmi, daha az bozulma" diye okunabiliyor ve bunun fiziksel bir karşılığı var.

## Zemini kurmak

İlk yazdığım kod modelle ilgili değildi. Rastgeleliğin tek bir yerden yönetilmesini sağlayan bir yapılandırma dosyasıydı. Projedeki bütün rastgele işlemler, eğitim-test bölmesi de çapraz doğrulama katmanları da model başlatmaları da, tohum değerini aynı yerden okuyor. Bunun sebebi performansı iyileştirmek değil; iki deney farklı skor ürettiğinde farkın yaptığım değişiklikten geldiğinden emin olmak. Tohumu sabitlemek varyansı yok etmez, sadece gizler; o yüzden beşinci fazdan itibaren tek bölme yerine çapraz doğrulamaya geçtim ve ortalamanın yanında yayılımı da raporlamaya başladım.

Bağımlılıkları sürüm numarasına kadar sabitledim ve Python'un en yeni sürümü yerine 3.11'i seçtim. Alternatifi 3.13'tü ve muhtemelen çalışırdı, ama XGBoost, LightGBM ve SHAP derlenmiş uzantılar içeriyor; yeni Python sürümlerinde hazır paketler gecikmeli geliyor ve gelmezse Windows'ta derleyici olmadan kurulum çöküyor. Projede 3.13'e özgü hiçbir özellik kullanmıyordum, yani risk alıp karşılığında bir şey kazanmıyordum. Üretim hattı yazılımını en yeni işletim sistemine ilk gün geçirmemek gibi bir karar.

GPU'yu hiç kullanmadım. Elimde bir GTX 1050 vardı ve boş durdu. Veri on bin satır ve on kolon, yani yaklaşık bir megabayt. Bu ölçekte ağaç toplulukları işlemcide saniyeler içinde eğitiliyor; veriyi ekran kartına kopyalamanın maliyeti eğitimden uzun sürerdi. Bu, beşinci fazda "tablo verisinde ağaçlar neden derin öğrenmeyi yener" sorusunun cevaplarından biri oldu ve ezberlenmiş değil ölçülmüş bir cevap olarak elimde kaldı.

## Veriyi tanımak ve ilk şüpheler

Keşifsel analiz üç şey gösterdi ve üçü de projenin geri kalanını şekillendirdi.

Birincisi, etiketin kendisi tutarsızdı. On sekiz satırda bir arıza modu yanıyor ama makine arızası sıfır yazıyordu, ve o on sekizin on sekizi de RNF'ti. Dokuz satırda ise tersi vardı: arıza deniyor ama hiçbir mod işaretli değil. Yirmi dört satırda birden fazla mod aynı anda yanıyordu. Bu bulgular o an sadece birer not gibi görünüyordu; onuncu fazda modelin kaçırdığı arızaların yarısının tam olarak bu bozuk satırlar olduğu ortaya çıkacaktı.

İkincisi, korelasyonlar zayıftı. Hedefle en yüksek korelasyona sahip sensör tork'tu ve o bile sadece 0.19. Buradan "model çıkmaz" sonucunu çıkarmak cazipti ama yanlış olurdu, çünkü Pearson korelasyonu yalnızca doğrusal ilişkiyi ölçer. Değişkenler birlikte etkili olabilir ve nitekim öyle oldu. Buna karşılık iki güçlü korelasyon vardı: mil devri ile tork arasında eksi 0.88, iki sıcaklık arasında artı 0.88. Birincisinin sebebi milin yaklaşık sabit güçte çalışması, ikincisinin sebebi ise iki sıcaklığın mevsimle birlikte kayması. İkinci gözlem şunu söylüyordu: bilgi sıcaklıkların kendisinde değil, aralarındaki farkta.

Üçüncüsü, bir fizik ipucuydu. Tork ile devri çarpıp iki pi bölü altmış ile ölçeklediğimde mekanik gücü elde ettim. AI4I dokümantasyonuna göre güç arızası, gücün 3500 ile 9000 watt bandının dışına çıkmasıyla tetikleniyordu. Bandın dışındaki satırları saydım: doksan beş. PWF etiketli satırları saydım: doksan beş. Sayılar birebir tutuyordu. Ama bunu o an kanıt saymadım, çünkü farklı doksan beş satır da "doksan beş ve doksan beş" verirdi. Kanıtı dördüncü fazda küme kesişimi alarak yapacaktım.

## Hiçbir şey yapmayan model

İkinci fazda kasten aptal bir model kurdum. Hiçbir özelliğe bakmayan bir sınıflandırıcıyı iki stratejiyle çalıştırdım. "En sık sınıfı tahmin et" stratejisi her parçaya sağlam dedi ve test setinde yüzde 96.60 doğruluk aldı. Stres varyantında aynı model yüzde 99.49 aldı. Yakaladığı arıza sayısı sıfırdı.

Ama asıl manşet ikinci stratejide çıktı. Sınıf oranlarına saygı göstererek rastgele tahmin eden model iki arıza yakaladı ve doğruluğu yüzde 96.60'tan yüzde 93.20'ye düştü. Yani bir şey yapan model, doğrulukta üç buçuk puan kaybetti. Bu tek gözlem, doğruluk metriğinin bu problemde neden kullanılamayacağını herhangi bir teorik açıklamadan daha net anlatıyor: doğruluğu optimize eden bir ekip, işe yarayan modeli aktif olarak reddeder.

F1 skorunu da ana metrik yapmadım. Alternatifi vardı ve yaygındır, ama F1 yanlış alarm ile kaçan arızaya eşit ağırlık verir. Kestirimci bakımda plansız duruş, gereksiz bir kontrolden çok daha pahalıdır ve eşit ağırlık veren bir metrik bu asimetriyi görmezden gelir. Sekizinci fazda açık bir maliyet matrisi tanımlayıp eşiği ona göre seçtim.

## Sızıntıyı bilerek üretmek

Üçüncü fazda iki deney yaptım ve ikisinde de model, bölme ve satırlar sabit kaldı; sadece modelin gördüğü bilgi değişti.

İlk deneyde arıza modu kolonlarını özellik olarak bıraktım. Sonuç: precision tam bir, recall 0.9706. Model hiç yanlış alarm vermedi. Bu bir başarı değil, alarm işaretidir; gerçek bir süreçte sıfır yanlış alarm neredeyse imkânsızdır. Sebep basitti: hedef değişken zaten bu kolonların birleşimiydi, model kuralı öğrenmiyor kopyalıyordu. Kolonları çıkarınca recall 0.2941'e düştü. Altmış sekiz puanın tamamı sızıntıydı.

İlginç bir ayrıntı da şuydu: cevap anahtarı elinde olan model bile yüzde yüz tutturamadı, iki arıza kaçırdı. Sebebi birinci fazda bulduğum RNF tutarsızlığıydı. Dahası, ağaç RNF kolonuna sıfır önem verdi; hedefle ilişkisiz olduğunu kendi keşfetti. Bu, ilk fazdaki tespitin bağımsız doğrulaması oldu.

İkinci deney bölme sızıntısıydı. Takım aşınması kümülatif bir değişken, yani satırlarda örtük bir sıra olabilirdi. Bunu varsaymak yerine ölçtüm: aşınma ardışık satırların yüzde 98.8'inde artıyor, yüzde 1.2'sinde düşüyordu ve o düşüşler yüz on dokuz takım değişimiydi. Yani ardışık satırlar aynı takım ömrüne ait, bağımsız değil. Rastgele bölme yerine sıraya göre bölünce recall 0.2941'den 0.2051'e indi.

Bu dokuz puanı raporlarken dikkatli olmam gerekti. Sıralı bölmede test setinde sadece otuz dokuz arıza kalmıştı, yani tek bir arıza 2.56 recall puanı ediyordu. Wilson güven aralıklarını hesapladım: rastgele bölme için 0.199 ile 0.411, sıralı bölme için 0.108 ile 0.355. Aralıklar büyük ölçüde çakışıyordu, yani dokuz puanlık düşüş gürültüden ayırt edilemiyordu. Bu yüzden sonucu recall düşüşü üzerinden değil, sıralı bölmenin ortaya çıkardığı başka bir bulgu üzerinden raporladım: ilk yüzde sekseninde arıza oranı yüzde 3.75'ken son yüzde yirmisinde yüzde 1.95'ti. Süreç zamanla kaymıştı ve rastgele bölme bunu tamamen gizliyordu. Bu bulgu üç yüz ve otuz dokuz arıza üzerinden sayıldığı için gürültü sorunu yaşamıyordu.

## Fiziği koda çevirmek

Dördüncü faz projenin dönüm noktasıydı. Üçüncü fazın temiz modeli altmış sekiz arızanın kırk sekizini kaçırıyordu ve sebebini biliyordum: makine tek bir okumadan bozulmuyor, okumaların kombinasyonundan bozuluyordu. Isı dağıtım arızası iki sıcaklığın farkına ve devre birlikte bakıyor, güç arızası tork ile devrin çarpımına, aşırı zorlanma arızası ise aşınma ile torkun çarpımına. Bir karar ağacı tek bölmede tek kolona eşik koyabilir; iki kolonun farkına bakan bir kuralı ancak çok basamaklı bölmelerle yaklaşık ifade edebilir ve derinlik bütçesinin tamamını buna harcar.

Üç kolon türettim. Tork çarpı devir çarpı iki pi bölü altmış, yani watt cinsinden mekanik güç. Proses sıcaklığı eksi ortam sıcaklığı, yani Kelvin cinsinden ısı gradyanı. Takım aşınması çarpı tork, yani kör takımın zorlanması. Bu üç formülün hiçbiri veriden çıkmıyor, mekanikten geliyor.

Burada ince bir sınır vardı ve onu bilerek çizdim: kolonu türetmek meşru, eşiği kodlamak sızıntıdır. Güç formülü evrensel fiziktir ve tork ile devir tahmin anında sensörden okunur. Ama koda "eğer güç 3500'ün altındaysa güç arızası vardır" yazsaydım, o sayı cevap anahtarından gelirdi ve üçüncü fazda reddettiğim hatanın aynısını yapmış olurdum. Kolonu ben türettim, eşiği model buldu. Dokümantasyondaki eşikler kodda var ama yalnızca doğrulama fonksiyonunda; model onları hiç görmedi.

Doğrulamayı sayı eşitliğiyle değil küme kesişimiyle yaptım. Güç bandı dışındaki satırlar ile güç arızası etiketli satırlar: doksan beş, doksan beş, kesişim doksan beş, iki yönlü fark sıfır. Isı gradyanı ve devir koşulu ile ısı dağıtım arızası: yüz on beş, yüz on beş, kesişim yüz on beş. Aşınma-tork çarpımı ile aşırı zorlanma arızası: doksan sekiz, doksan sekiz, kesişim doksan sekiz. Üçünde de özdeş kümeler.

Sonra aynı ağacı, aynı bölmeyi ve aynı tohumu kullanarak modeli tekrar çalıştırdım. Recall 0.2941'den 0.8235'e, precision 0.7692'den 0.9492'ye çıktı. Kaçan arıza sayısı kırk sekizden on ikiye indi. Ağaç dikkatinin yüzde 67.1'ini bir saat önce var olmayan üç kolona kaydırmıştı.

Precision ile recall'ın aynı anda artması dikkate değerdi, çünkü normalde ters çalışırlar. Sebep şuydu: bu bir eşik ayarı değildi. Model daha cesur tahmin etmiyordu, doğru soruyu soruyordu. Ayrım gücü gerçekten arttığında iki metrik birlikte yükselir.

Bir de beklemediğim bir şey oldu: mil devrinin önemi yüzde 11.9'dan yüzde 27.0'a çıktı. Kolon aynı kolondu. Sebebi, ısı dağıtım arızasının iki koşullu olması: sıcaklık farkı hazır gelince ağaç bir bölme harcayıp hemen devre geçebiliyordu. Yani yeni bir özellik, mevcut özellikleri de daha kullanışlı hale getirebiliyor.

Kalan on iki kaçağı moda göre ayırdım. Güç arızasında recall bir, aşırı zorlanmada bir, ısı dağıtımında 0.97. Kaçanların dokuzu takım aşınması arızasıydı. Bunu kovalamadım ve sebebini yazdım: AI4I'de takım, iki yüz ile iki yüz kırk dakika arasında rastgele seçilen bir anda bozuluyor. Bandı öğrensen bile içindeki kurayı öğrenemezsin. Kovalasaydım eğitim skorunu şişirir, sahada aynı sonucu alırdım.

## Merdiveni tırmanmak ve boş dönmek

Beşinci fazda dört model ailesi denedim: lojistik regresyon, random forest, XGBoost ve LightGBM. Ölçüm aletini de değiştirdim; tek bölme yerine beş katlı katmanlı çapraz doğrulamaya geçtim. Beş kat seçtim çünkü üç yüz otuz dokuz arıza beşe bölününce her ayrılmış parçaya yaklaşık altmış sekiz arıza düşüyor ve bu, önceki fazlardaki test seti boyutuyla aynı. On kat olsaydı otuz dört kalırdı ve tek bir arıza üç recall puanı ederdi.

Sonuç şuydu: merdiven neredeyse hiçbir şey kazandırmadı. En iyi PR-AUC random forest'ta, 0.904; referans karar ağacınınki 0.865. Bu fark fold yayılımından büyük olduğu için gerçek ama küçük. Recall'da ise random forest 0.814, karar ağacı 0.823 aldı; fark yayılımın içinde kaldığı için fark yok, üstelik yirmi üç kat hesap maliyeti karşılığında yok. Dördüncü fazda üç kolon eklemek recall'ı 0.294'ten 0.824'e taşımıştı. Özellik mühendisliği, model seçiminden yaklaşık elli kat daha çok kazandırmıştı ve bunu tahmin etmedim, ölçtüm.

Lojistik regresyon çöktü: recall 0.233, PR-AUC 0.495. Bu bir kod hatası değildi, modelin ifade gücünün sınırıydı. Güç arızası, gücün 3500'ün altında veya 9000'in üstünde olmasıyla tetikleniyor. Doğrusal bir sınır her değişken için tek bir katsayı öğrenir, yani tek bir yön; hem çok az hem çok çok diyemez. Ağaçlar bunu iki bölmeyle çözüyor. Basit modeli atlayıp doğrudan XGBoost'a gitseydim 0.87 alır ve neden aldığımı bilmezdim.

Boosting'in bagging'i yenememesi de öğreticiydi. Boosting'in üstünlüğü karmaşık ve öğrenilecek yapısı bol problemlerde çıkar; dördüncü fazdan sonra bu problem kolaylaşmıştı ve kalan hatanın büyük kısmı zaten öğrenilemez gürültüydü.

## Metriklerin yalanı

Altıncı fazda aynı modeli iki rejimde çalıştırıp PR ve ROC eğrilerini yan yana koydum. Arıza oranı yüzde 3.4'ten yüzde 0.515'e inince recall 0.8235'ten 0.5000'e düştü; model artık her iki arızadan birini kaçırıyordu. PR-AUC 0.8901'den 0.6530'a indi. ROC-AUC ise 0.9781'den 0.8799'a, yani gerçek kaybın yaklaşık üçte biri kadar. Ve 0.88 sayısı hiçbir raporda alarm zili çaldırmaz.

Mekanizma paydalarda. Yanlış pozitif oranının paydası tüm sağlam satırlar ve stres rejiminde bin dokuz yüz otuz üç tane var; yirmi fazladan yanlış alarm o oranı 0.01 oynatır, ROC eğrisinde neredeyse görünmez. Precision'ın paydası ise verilen alarmlar; aynı yirmi alarm precision'ı yerle bir eder. En temiz anlatımı tabanlarda: yazı-tura atan bir model arıza ne kadar nadir olursa olsun ROC-AUC 0.5 alır, taban hiç kımıldamaz. Aynı model PR-AUC'de arıza oranının kendisini alır. PR-AUC referansını problemle birlikte taşır, ROC-AUC taşımaz.

Bu fazda bir dürüstlük notu düşmem gerekti: stres rejiminde test setinde sadece on arıza vardı, yani tek arıza on recall puanı ediyordu. Trendin yönüne güveniyorum, ondalık basamağa güvenmiyorum.

## Reçeteyi denemek ve reddetmek

Yedinci fazda dengesizlik için önerilen iki standart yöntemi denedim: sınıf ağırlıklandırma ve SMOTE. İkisi de recall'ı fold yayılımından küçük miktarda artırdı, yani ölçülemeyecek kadar az. Buna karşılık kayıpları büyük ve gerçekti: SMOTE precision'ı 0.968'den 0.654'e düşürdü ve stres rejiminde PR-AUC'yi 0.698'den 0.321'e çökertti.

SMOTE'un neden bu kadar kötü olduğunu aritmetik açıklıyor. Stres rejiminde her katmanda kırk gerçek arızadan yaklaşık yedi bin yedi yüz sentetik arıza üretmesi gerekiyordu; yani modelin gördüğü pozitiflerin yüzde 99.5'i uydurmaydı. Üstelik bu veride tork ile devir eksi 0.88 korelasyonlu, çünkü mil sabit güçte çalışıyor. İki gerçek arıza arasında doğrusal interpolasyon yapmak, makinenin hiç üretmediği bir tork-devir kombinasyonu veriyor. Kâğıtta pozitif sayısı artıyor, sahada model olmayan bir bölgeyi öğreniyor. Alternatifi SMOTE'u kullanıp metriklerin bir kısmının iyileşmesine sevinmekti; ölçüp reddetmeyi seçtim.

Sınıf ağırlıklandırmanın işe yaramamasının sebebi farklıydı. O yöntem, modelin azınlık sınıfını göremediği durumlarda işe yarar. Dördüncü fazdan sonra model azınlık sınıfını gayet iyi görüyordu. Kalan sorun görememek değil, nerede keseceğini bilmemekti ve onun aracı ağırlık değil eşiktir.

Kalibrasyonu da ölçtüm çünkü bir sonraki fazda eşiği olasılık ekseni üzerinde seçecektim. Ham random forest 0.00674 Brier skoruyla en iyisi çıktı; isotonic 0.00825, sigmoid 0.00969. Yani kalibrasyon sarmalayıcıları durumu kötüleştirdi. Sebebi, sarmalayıcının temel modeli verinin üçte ikisiyle eğitmesi; üç yüz otuz dokuz pozitifle bu takas zarar ediyor. Ayrıca random forest üç yüz ağacın oy oranını verdiği için zaten makul kalibre bir model.

## Projenin kalbi

Sekizinci faza kadar sekiz fazdır 0.5 eşiğini kullanmıştım ve bir kez bile savunmamıştım. Savunulamaz da: 0.5, kimse bir şey söylemediğinde kütüphanenin yaptığı şey ve örtük olarak yanlış alarm ile kaçan arızanın aynı maliyette olduğunu iddia ediyor.

Açık bir maliyet matrisi tanımladım. Kaçan arıza elli, yanlış alarm bir. Mutlak sayılar önemli değil ve doğru olduklarını da iddia etmiyorum; optimizasyona giren tek şey oran. Elliye bir, kabaca sekiz saatlik plansız hat duruşunu on dakikalık bir kontrole karşı koyuyor. Bu bir model parametresi değil, işletme girdisi.

Eşiği 0.01'den 0.99'a taradım ve her noktada gerçek maliyeti ayrılmış veride ölçtüm. Olasılıkları çapraz doğrulama tahminleriyle ürettim, çünkü eşiği modelin ezberlediği satırlar üzerinden seçmek üçüncü fazın hatasının başka kılıktaki hali olurdu.

Optimum eşik 0.03 çıktı. Varsayılan 0.5 ile karşılaştırınca kırk dört arıza daha yakalanıyor, altı yüz yirmi beş fazladan yanlış alarm veriliyor ve toplam maliyet 3159'dan 1584'e, yani yüzde 49.9 düşüyor. Model bu iki nokta arasında bayt bayt aynı; sadece karar kuralı değişti.

Kalibre bir model için teorik optimum, yanlış pozitif maliyetinin toplam maliyete oranıdır ve 0.0196 veriyor. Ampirik tarama 0.0300 buldu. İki bağımsız yöntemin bu kadar yakın sonuç vermesi, modelin olasılık ekseninin düzgün davrandığının kanıtı; yedinci fazdaki Brier bulgusunun ikinci doğrulaması.

Oran bir varsayım olduğu için duyarlılık analizi yaptım. Onda birde eşik 0.34, ellide 0.03, iki yüzde 0.01. Eşik bir büyüklük mertebesi oynuyor ama recall sadece 0.85'ten 0.96'ya çıkıyor. Bu, azalan getirinin eğri olarak çizilmiş hali: ilk arızaları yakalamak ucuz, sonuncuları pahalı.

Bir sınırlılığı da gizlemek yerine yazdım. Maliyet modelim doğrusal, yani altı yüzüncü yanlış alarmı birincisiyle aynı fiyatlıyor. Gerçekte alarm yorgunluğu var ve bu maliyet süperdoğrusaldır. Dokuz bin altı yüz altmış bir sağlam parçada altı yüz otuz dört alarm, her on beş parçadan birine kontrol demek. Kapasite verisi olsaydı vardiya başına alarm bütçesi kısıtı ekleyip eşiği o kısıt altında optimize ederdim.

## Tahminden iş emrine

Dokuzuncu fazda SHAP kullandım. Özellik önem tablosu her satır için aynı cevabı verir ve bakım ekibine önündeki makine hakkında hiçbir şey söyleyemez. SHAP ise tek bir tahmini katkılarına ayırıyor ve katkıların toplamı, taban değerle birlikte, modelin çıktısına tam olarak eşit oluyor. Bu toplanabilirlik özelliği açıklamayı yorum olmaktan çıkarıp muhasebeye çeviriyor.

Gerçek bir arıza satırını inceledim. Model 0.997 vermişti. Kararın yüzde elli altısı güce atfedildi: güç 2753 watt okumuştu, alt bandın yedi yüz elli watt altında. Onu tork ve devir izledi; tork 9.7 Newton-metre ile anormal düşük, devir 2710 ile anormal yüksekti. Fiziksel okuması şu: mil hızlı dönüyor ama iş yapmıyor, kesici malzemeye girmiyor. Bakım ekibine söylenecek cümle "bu makine bozulacak" değil, "kesici takımı ve besleme parametrelerini kontrol et". Birincisi inanılıp inanılmayacak bir tahmin, ikincisi uygulanabilir bir iş emri.

Küresel SHAP sıralaması, dördüncü fazdaki özellik önem sıralamasıyla uyuşmadı. Sıcaklık farkı impurity tabanlı ölçüde dördüncü sıradayken yüzde 4.6 alıyordu, SHAP'te birinci sıraya çıktı. Sebebini de biliyorum: ısı dağıtım arızası en yaygın mod ve sıcaklık farkı onun karar kolonu; çok sayıda satırı azar azar oynatıyor. Impurity tabanlı önem ise kökteki büyük bölmeleri kayırıyor. İkisi çeliştiğinde SHAP'e güveniyorum, çünkü modelin çıktı biriminde ve satır bazında ölçüyor.

## Neyin kovalanmayacağını bilmek

Onuncu fazda kaçan arızaları arıza moduna göre ayırdım, çünkü toplam recall tek başına kalan hatanın kapatılmaya değer olup olmadığını söylemiyor.

Maliyet optimumu eşiğinde sonuç şuydu: ısı dağıtım arızasında yüz on beşte yüz on beş, güç arızasında doksan beşte doksan beş, aşırı zorlanmada doksan sekizde doksan sekiz. Fiziksel olarak belirlenmiş üç modun üçünde de recall bir. Eşik değişimi takım aşınması arızasını 0.065'ten 0.783'e taşıdı; onun sinyali zayıf olduğu için eşiğe en duyarlı mod oydu.

Kaçan on dokuz arızanın onu takım aşınması, dokuzu ise hiçbir arıza modu işaretlenmemiş satırlardı. İkisi de indirgenemez. Takım aşınmasında bozulma anı kura ile belirleniyor; modu olmayan satırların sensör değerlerinde onları arıza yapan hiçbir şey yok. Yani kapatılabilecek bir açık kalmadı ve bunu bilmek, kovalamayı bırakmak için gereken şey.

Rastgele arıza modunu kapsam dışı bıraktım ve gerekçesini iki ayakta yazdım. Tanım gereği sensörlerden bağımsız üretiliyor, yani hiçbir özellik onun hakkında bilgi taşımıyor. Üstelik etiket de tutarsız: on dokuz satırın on sekizi arıza bile sayılmamış. Modelin bu satırlara sağlam satırlarla neredeyse aynı skoru vermesi hata değil, girdilerde olmayan bir sinyal için doğru davranış.

Son olarak yanlış alarmların profiline baktım, çünkü altı yüz otuz dört alarm veren bir sistemin savunulabilir olması gerekiyordu. Alarm veren sağlam satırların aşınma-tork çarpımı ortalaması 7224, sessiz kalanların 4002. Yani bunlar rastgele satırlar değil, arıza bölgesine kıl payı yakın vakalar. Teknisyen gittiğinde bozuk bir makine bulmayacak ama sınırda çalışan bir makine bulacak. Bu, süreç kontrolündeki kontrol limiti ile spesifikasyon limiti farkına benziyor: parça hâlâ toleransta olabilir ama süreç kaymaya başlamıştır.

## Sınırlar ve sahaya alma

Bu projenin en büyük sınırlılığı verinin sentetik olması. Arıza modları belli kurallarla üretildiği için fiziksel özellik türetmek beklenmedik ölçüde etkili oldu; gerçek bir tezgâhta sınırlar bu kadar keskin olmaz. İkinci sınırlılık, gerçek zaman damgasının olmaması. Satır sırasından örtük bir zaman çıkarabildim ama gerçek bir kestirimci bakım projesinde etiket "şu anda arızalı" değil "önümüzdeki birkaç saat içinde arızalanacak" biçiminde yazılır ve o ufuk, yedek parça tedarik süresi ile bozulmanın sensörde görünür olduğu süre arasında bir yerde, operasyonla birlikte belirlenir. Üçüncüsü, rastgele arıza modunun tanımı gereği öngörülemez olması.

Sahaya alsam gölge modda başlatırdım. Model üretim hattına bağlanır, olasılıkları ve alarmları üretir ama hiçbir alarm operatöre gitmez; sadece kaydedilir. Üç dört hafta sonra elimde gerçekleşen arızalarla modelin uyarıları yan yana durur ve o zaman şu soru cevaplanabilir hale gelir: model gerçekten önceden uyardı mı, yoksa arızayla aynı anda mı fark etti. Aynı dönemde maliyet oranını bakım ve üretim ekipleriyle birlikte netleştirir, eşiği o gerçek sayılarla yeniden seçerdim.

İkinci aşamada alarmları sınırlı bir gruba, tercihen tek bir hattın vardiya amirine açardım ve her alarm için geri bildirim toplardım: gidildi mi, ne bulundu, müdahale edildi mi. Bu geri bildirim iki işe yarar; hem modelin gerçek precision'ını ölçer hem de yanlış alarmların gerçekten kıl payı vakalar olup olmadığını sahada doğrular. Alarm sayısı vardiyanın kaldırabileceğinden fazlaysa eşiği maliyet optimumundan kapasite kısıtına çekerdim, çünkü cevaplanamayan alarm alarm sayılmaz.

Üçüncü aşamada, ancak gölge dönemi ve pilot dönemi beklentiyi tuttuysa yaygınlaştırırdım. Ve yaygınlaştırırken izlenecek şeyin model skoru değil süreç kayması olduğunu baştan söylerdim; üçüncü fazda ölçtüğüm gibi arıza oranı zaman içinde yarıya inebiliyor ve o durumda eşiğin de modelin de yeniden gözden geçirilmesi gerekir.

---

# Mülakatta Sorulabilecek On Beş Soru ve Cevapları

**1. Modeliniz yüzde 99 doğruluk aldı. İyi bir sonuç değil mi?**

Tek başına hiçbir şey söylemez. Bu veride arıza oranı yüzde 3.39; "her parça sağlamdır" diyen ve hiçbir şey öğrenmeyen bir model yüzde 96.60 alıyor, nadir arıza rejiminde ise yüzde 99.49. Yakaladığı arıza sıfır. Dahası, rastgele tahmin eden ve iki arıza yakalayan bir model doğrulukta üç buçuk puan kaybediyor. Yani doğruluğu optimize eden biri, işe yarayan modeli aktif olarak reddeder. Ben karmaşıklık matrisine mutlak sayılarla, recall'a ve PR-AUC'ye bakarım.

**2. Veri sızıntısı nedir ve nasıl fark edersiniz?**

Modelin eğitimde gördüğü ama tahmin anında elde olmayacak bilgidir. Kod hatası değildir; sayılar doğru hesaplanır ama ölçülen şey sahada tekrarlanamaz. Fark etme yollarım: beklenmedik yüksek skoru ilk olarak sızıntı hipotezi sayarım, özellik önemlerine bakarım, ve her kolon için "bu ne zaman doluyor" sorusunu sorarım. Bu projede arıza modu kolonlarını özellik bıraktığımda precision tam bir çıktı ve dikkatin yüzde yüzü o kolonlardaydı; onlar arıza sonrası doldurulan teşhis kayıtları. Çıkarınca recall 0.97'den 0.29'a düştü.

**3. Neden rastgele eğitim-test bölmesi her zaman doğru değildir?**

Satırların bağımsız olduğunu varsayar. Bu veride değiller: takım aşınması ardışık satırların yüzde 98.8'inde artıyor, yüzde 1.2'sinde düşüyor ve o düşüşler takım değişimleri. Yani komşu satırlar aynı takım ömrüne ait. Sıraya göre bölünce recall düştü, ama asıl bulgu o değildi: sıralı bölme, rastgele bölmenin tamamen gizlediği bir süreç kaymasını gösterdi. İlk yüzde seksende arıza oranı yüzde 3.75, son yüzde yirmide yüzde 1.95. Veride zaman, makine kimliği veya üretim lotu gibi bir gruplama varsa bölme o grubu bölmeyecek şekilde yapılır.

**4. Özellik mühendisliği yaptınız mı, örnek verir misiniz?**

Alan bilgisinden üç değişken türettim: torktan ve devirden mekanik gücü, iki sıcaklıktan gradyanı, aşınma ile torktan zorlanma göstergesini. Aynı model, aynı bölme ve aynı tohumla recall 0.29'dan 0.82'ye, precision 0.77'den 0.95'e çıktı. Model daha akıllı olmadı; ona makinenin nasıl bozulduğunu anlatan dili verdim. Türetmenin doğruluğunu da varsaymadım: güç bandı kuralının seçtiği doksan beş satır ile güç arızası etiketli doksan beş satırın aynı satırlar olduğunu kesişim alarak doğruladım, iki yönlü fark sıfır.

**5. Arıza eşiklerini dokümantasyondan biliyordunuz. Bu sızıntı değil mi?**

Sınır şurada: kolonu türetmek meşru, eşiği kodlamak sızıntı. Güç formülü evrensel fiziktir ve girdileri tahmin anında sensörden okunur. Ama "eğer güç 3500'ün altındaysa arıza" yazsaydım o sayı cevap anahtarından gelirdi. Kodumda dokümantasyon eşikleri var ama yalnızca doğrulama fonksiyonunda; model onları hiç görmedi ve kesme noktalarını kendi buldu.

**6. Tablo verisinde ağaç toplulukları neden derin öğrenmeyi yener?**

Az veriyle çalışır, ölçekleme gerektirmez, eksik veriye dayanıklıdır, kategorik değişkeni doğal işler, GPU gerektirmez ve en önemlisi açıklanabilir. Bu projede on bin satır ve üç yüz otuz dokuz pozitif vardı; en yavaş model katman başına 0.7 saniye sürdü ve ekran kartı hiç kullanılmadı. Ölçekleme gereken tek model lojistik regresyondu ve o da en kötüsü çıktı. Açıklanabilirlik ise fabrikada belirleyici: model bir tahmin makinesi değil, kök-neden işaretçisidir. Süreç mühendisine nereye müdahale edeceğini söylemeyen model işe yaramaz. Ayrıca derin öğrenmenin asıl gücü olan hiyerarşik temsil çıkarma işini burada zaten elle yaptım.

**7. Neden lojistik regresyonu da denediniz, sonucu kötüyse?**

Çünkü kötü sonuç da bilgi. Recall 0.233 ve PR-AUC 0.495, problemin doğrusal olmadığını kanıtladı. Güç arızası 3500'ün altında veya 9000'in üstünde diye tanımlı; doğrusal bir sınır her kolon için tek katsayı öğrenir, yani tek yön, hem çok az hem çok çok diyemez. Doğrudan XGBoost'a gitseydim 0.87 alır ve neden aldığımı bilmezdim.

**8. ROC-AUC ne zaman yanıltır?**

Sınıflar dengesiz olduğunda. Sebebi paydalar: yanlış pozitif oranı tüm sağlam satırlara bölünür ve onlar binlercedir, o yüzden yüzlerce yanlış alarm bile oranı kıpırdatmaz. Precision ise verilen alarmlara bölünür ve hepsini hisseder. Bu projede arıza oranını yüzde 3.4'ten yüzde 0.5'e indirdiğimde recall 0.82'den 0.50'ye düştü, model her iki arızadan birini kaçırıyordu; ROC-AUC ise sadece 0.978'den 0.880'e indi. En temiz ayrım tabanlarda: yazı-tura her zaman ROC-AUC 0.5 alır, PR-AUC'de ise arıza oranının kendisini alır.

**9. Dengesiz veriyle nasıl başa çıkarsınız?**

Önce ölçer, sonra karar veririm, ve bu projede ölçüm sonucu "hiçbiri" çıktı. Sınıf ağırlıklandırma ve SMOTE'un recall kazançlarının ikisi de fold yayılımının içinde kaldı, yani ölçülemedi; buna karşılık SMOTE precision'ı 0.97'den 0.65'e düşürdü ve nadir rejimde PR-AUC'yi 0.70'ten 0.32'ye çökertti. Bu yöntemler modelin azınlık sınıfını göremediği durumlarda işe yarar; benim modelim onu gayet iyi görüyordu, sorun nerede keseceğini bilmemekti ve onun aracı eşiktir.

**10. SMOTE hakkında ne düşünüyorsunuz?**

Fiziksel verilerde dikkatli olurum. İki gerçek azınlık örneği arasında doğrusal interpolasyon yapar; değişkenler arasında fiziksel bir kısıt varsa var olamayacak durumlar üretir. Bu veride tork ile devir eksi 0.88 korelasyonlu çünkü mil sabit güçte çalışıyor, dolayısıyla iki arızanın orta noktası makinenin hiç üretmediği bir çalışma noktası. Ölçtüm: stres rejiminde kırk gerçek arızadan yaklaşık yedi bin yedi yüz sentetik üretmesi gerekti, yani pozitiflerin yüzde 99.5'i uydurma. Ayrıca teknik bir tuzağı var: mutlaka boru hattı içinde, bölmeden sonra çalışmalı, yoksa test satırlarının sentetik komşuları eğitime düşer.

**11. Kalibrasyon nedir ve ne zaman gerekir?**

Modelin ürettiği sayının gerçekten olasılık olmasıdır; 0.7 dediği satırların gerçekten yüzde yetmişi arızalanmalı. Reliability diagram ile ölçülür, Brier skoru tek sayıya indirir. Gerekli olduğu yerler: olasılığı bir insana söyleyeceksen, beklenen alarm sayısı tahmin edeceksen, eşiği başka bir hatta veya döneme taşıyacaksan. Gerekli olmadığı yer: eşiği ampirik olarak ayrılmış veride tarayarak seçiyorsan, çünkü kalibrasyon monoton bir dönüşümdür ve sıralamayı değiştirmez. Bu projede kalibrasyon sarmalayıcıları Brier'i kötüleştirdi çünkü temel modeli verinin üçte ikisiyle eğitiyorlar.

**12. Eşiği nasıl seçersiniz?**

Eşik bir model kararı değil, maliyet kararıdır. Ben olasılığı üretirim, eşiği kalite ve üretim ekibiyle maliyet üzerinden belirleriz. Kaçan arızayı elli, yanlış alarmı bir olarak fiyatladım ve eşiği 0.01'den 0.99'a tarayıp gerçek maliyeti ayrılmış veride ölçtüm. Optimum 0.03 çıktı; varsayılan 0.5 ile karşılaştırınca kırk dört arıza daha yakalanıyor, altı yüz yirmi beş fazladan alarm veriliyor ve toplam maliyet yarıya iniyor. Oran bir varsayım olduğu için duyarlılık analizi de yaptım: onda birde eşik 0.34, iki yüzde 0.01.

**13. Precision 0.34'e düştü. Bu kötü bir model değil mi?**

Model değişmedi; precision düşüşü tamamen benim seçtiğim karar kuralının sonucu ve bilinçli. Elliye bir varsayımı altında altmış üç kaçan arıza, altı yüz otuz dört yanlış alarmdan çok daha pahalı. İyi model tanımını metrik güzelliği değil iş sonucu belirler. Ama bir sınırlılığı da raporluyorum: maliyet modelim doğrusal, oysa alarm yorgunluğu gerçek ve o maliyet süperdoğrusal.

**14. Modelinizi nasıl açıklıyorsunuz?**

İki katmanda. Küresel katmanda SHAP ortalamalarıyla hangi değişkenlerin arızayı sürüklediğini, yerel katmanda tek bir tahmini katkılarına ayırarak. Bir örnek: model bir satıra 0.997 verdi ve kararın yüzde elli altısı güce atfedildi; güç 2753 watt okumuştu, alt band 3500. Bakım ekibine söylediğim cümle "bu makine bozulacak" değil, "kesici takımı ve besleme parametrelerini kontrol et". SHAP'i tercih etmemin teknik sebebi toplanabilirlik: taban değer artı katkılar, modelin çıktısına tam olarak eşit. Bu, açıklamayı yorum olmaktan çıkarıp muhasebeye çeviriyor.

**15. Modelinizin sınırı nerede?**

Kaçan arızaları arıza moduna göre ayırdım. Fiziksel olarak belirlenmiş üç modda recall bir: ısı dağıtımında yüz on beşte yüz on beş, güçte doksan beşte doksan beş, aşırı zorlanmada doksan sekizde doksan sekiz. Kaçan on dokuz arızanın onu takım aşınması, dokuzu ise hiçbir modu işaretlenmemiş satırlar. İkisi de indirgenemez: takım iki yüz ile iki yüz kırk dakika arasında kura ile bozuluyor, modu olmayan satırların sensör değerlerinde onları arıza yapan hiçbir şey yok. Yani kapatılabilecek bir açık kalmadı. Rastgele arıza modunu da kapsam dışı bıraktım; sensörlerden bağımsız üretiliyor ve on dokuz satırının on sekizi arıza bile sayılmamış. Bir modelin öğrenilemez kısmını tanımlayıp kovalamamak, o kısmı kovalayıp eğitim skorunu şişirmekten daha değerli.
