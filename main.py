from src.processing import DataProcessor


processor = DataProcessor(
    'data/final_tickets_orders_df.csv',
    'data/final_tickets_events_df.csv',
    'data/final_tickets_tenge_df.csv'
)

df_raw = processor.load_data()
processor.load_currency()
df = processor.preprocess()
user_profile = processor.build_user_profile()

retention = user_profile['is_two'].mean()
events = processor.retention_by_segment('event_first_order')
devices = processor.retention_by_segment('device_first_order')
regions = processor.retention_by_segment('region_first_order')
weekdays = processor.retention_by_segment('first_order_weekday')
hypothesis_1 = processor.check_hypothesis_1()
hypothesis_2, region_corr = processor.check_hypothesis_2()
corr = processor.correlation_analysis()

median_revenue = user_profile.groupby('is_two')['avg_revenue'].median()

sport = hypothesis_1.loc['спорт', 'return_rate']
concerts = hypothesis_1.loc['концерты', 'return_rate']
top_regions = hypothesis_2.loc['топ-10 регионов', 'return_rate']
other_regions = hypothesis_2.loc['остальные регионы', 'return_rate']


print('=' * 60)
print('ИТОГОВЫЙ ОТЧЁТ: лояльность пользователей Яндекс Афиши')
print('=' * 60)

print()
print('1. ДАННЫЕ И ПОДГОТОВКА')
print(f'Период: с {df["order_dt"].min().date()} по {df["order_dt"].max().date()}')
print(f'Заказов в выгрузке: {df_raw.shape[0]}, после очистки: {df.shape[0]}')
print('- выручка в тенге переведена в рубли (revenue_rub)')
print('- удалены возвраты (выручка <= 0)')
print('- выручка отфильтрована по 99-му перцентилю')
print('- исключены аномальные пользователи (заказов больше 95-го перцентиля)')

print()
print('2. ПОЛЬЗОВАТЕЛИ')
print(f'Пользователей в выборке: {user_profile.shape[0]}')
print(f'Медиана числа заказов: {user_profile["total_orders"].median():.0f}')
print(f'Доля вернувшихся (2+ заказа): {retention:.1%}')

print()
print('3. RETENTION ПО СЕГМЕНТАМ')
print()
print(events.round(3))
print()
print(devices.round(3))
print()
print('Топ-10 регионов:')
print(regions.head(10).round(3))

print()
print('4. ПРОВЕРКА ГИПОТЕЗ')
print(f'Гипотеза 1: спорт {sport:.1%} против концертов {concerts:.1%}')
if sport > concerts:
    print('-> подтвердилась: после спорта возвращаются чаще')
else:
    print('-> не подтвердилась: после спорта возвращаются реже')

print(f'Гипотеза 2: топ-10 регионов {top_regions:.1%} против остальных {other_regions:.1%}')
print(f'Корреляция размера региона и доли возвратов: {region_corr:.2f}')
if top_regions > other_regions:
    print('-> подтвердилась частично: в крупных регионах возвращаются чаще, но связь слабая')
else:
    print('-> не подтвердилась')

print()
print('5. ВЫРУЧКА И ДЕНЬ НЕДЕЛИ')
print(f'Медиана средней выручки: 1 заказ - {median_revenue[False]:.0f} руб., '
      f'2+ заказа - {median_revenue[True]:.0f} руб.')
print(f'Доля возвратов по дню недели первого заказа: '
      f'от {weekdays["return_rate"].min():.1%} до {weekdays["return_rate"].max():.1%}')
print('-> день недели на возврат практически не влияет')

print()
print('6. КОРРЕЛЯЦИЯ phi_k С ЧИСЛОМ ЗАКАЗОВ')
print(corr.sort_values(by='orders_segment', ascending=False).round(2))
print('-> сильнее всего связаны поведенческие признаки: время между заказами и средняя выручка')
print('-> признаки первого заказа (регион, устройство, тип мероприятия) почти не связаны')

print()
print('7. РЕКОМЕНДАЦИИ')
print('1) Удержание строить на поведении после первой покупки: подборки и напоминания в первые недели')
print('2) Основной поток - мобильные пользователи и концерты: даже небольшой рост возвратов даст большой эффект')
print('3) Спорт - самое низкое удержание: предлагать после него другие категории')
print('4) Пользователи с очень дешёвым первым заказом возвращаются реже: нужен отдельный сценарий вовлечения')
print('5) Театр и выставки - аудитория с высоким удержанием: программы лояльности и абонементы')
print('6) Аккаунты с тысячами заказов искажают метрики: выявлять их и исключать из аналитики')
