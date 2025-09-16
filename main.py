import pandas as pd
import numpy as np
import json
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import NearestNeighbors
import folium
from folium.plugins import MarkerCluster
import streamlit as st
from geopy.distance import geodesic
import pickle
import os

class UniversityMLProject:
    def __init__(self):
        self.df = None
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.university_data = None
        self.nn_model = None
        
    def load_and_process_data(self, json_file='output.json'):
        print("Loading and processing data...")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        processed_data = []
        
        for item in data:
            try:
                address = item['address']
                if item['geocode']['status'] == 'OK' and len(item['geocode']['results']) > 0:
                    result = item['geocode']['results'][0]
                    
                    lat = result['geometry']['location']['lat']
                    lng = result['geometry']['location']['lng']
                    place_id = result['place_id']
                    location_type = result['geometry']['location_type']
                    
                    components = result['address_components']
                    
                    il = None
                    ilce = None
                    mahalle = None
                    postal_code = None
                    
                    for comp in components:
                        if 'administrative_area_level_1' in comp['types']:
                            il = comp['long_name']
                        elif 'administrative_area_level_2' in comp['types']:
                            ilce = comp['long_name']
                        elif 'administrative_area_level_4' in comp['types']:
                            mahalle = comp['long_name']
                        elif 'postal_code' in comp['types']:
                            postal_code = comp['long_name']
                    
                    types = result.get('types', [])
                    has_university = 'university' in types
                    has_establishment = 'establishment' in types
                    has_poi = 'point_of_interest' in types
                    
                    uni_name = address.split(',')[0] if ',' in address else address
                    
                    is_private = any(word in uni_name.lower() for word in ['özel', 'vakıf', 'foundation'])
                    
                    processed_data.append({
                        'university_name': uni_name,
                        'original_address': address,
                        'lat': lat,
                        'lng': lng,
                        'place_id': place_id,
                        'location_type': location_type,
                        'il': il,
                        'ilce': ilce,
                        'mahalle': mahalle,
                        'postal_code': postal_code,
                        'has_university': has_university,
                        'has_establishment': has_establishment,
                        'has_poi': has_poi,
                        'is_private': is_private,
                        'types_count': len(types)
                    })
            except Exception as e:
                print(f"Error processing {address}: {e}")
                continue
        
        self.df = pd.DataFrame(processed_data)
        print(f"Successfully processed {len(self.df)} universities!")
        return self.df
    
    def create_features_for_ml(self):
        print("Preparing features for ML...")
        
        if self.df is None:
            raise ValueError("Data must be loaded first!")
        
        region_mapping = {
            'İstanbul': 'Marmara',
            'Ankara': 'İç Anadolu', 'Konya': 'İç Anadolu', 'Kayseri': 'İç Anadolu',
            'İzmir': 'Ege', 'Manisa': 'Ege', 'Aydın': 'Ege', 'Muğla': 'Ege',
            'Antalya': 'Akdeniz', 'Adana': 'Akdeniz', 'Mersin': 'Akdeniz',
            'Samsun': 'Karadeniz', 'Trabzon': 'Karadeniz', 'Ordu': 'Karadeniz',
            'Erzurum': 'Doğu Anadolu', 'Van': 'Doğu Anadolu', 'Malatya': 'Doğu Anadolu',
            'Gaziantep': 'Güneydoğu Anadolu', 'Diyarbakır': 'Güneydoğu Anadolu'
        }
        
        self.df['bolge'] = self.df['il'].map(region_mapping).fillna('Diğer')
        
        buyuk_sehirler = ['İstanbul', 'Ankara', 'İzmir', 'Bursa', 'Antalya', 'Adana', 'Konya']
        self.df['sehir_kategori'] = self.df['il'].apply(
            lambda x: 'Büyükşehir' if x in buyuk_sehirler else 'İl'
        )
        
        self.df['lat_kategori'] = pd.cut(self.df['lat'], bins=5, labels=['Güney', 'Güney-Orta', 'Orta', 'Orta-Kuzey', 'Kuzey'])
        self.df['lng_kategori'] = pd.cut(self.df['lng'], bins=5, labels=['Batı', 'Batı-Orta', 'Orta', 'Orta-Doğu', 'Doğu'])
        
        return self.df
    
    def train_ml_model(self, target_column='bolge'):
        print("Training ML model...")
        
        if self.df is None:
            raise ValueError("Features must be prepared first!")
        
        feature_columns = ['lat', 'lng', 'has_university', 'has_establishment', 
                          'has_poi', 'is_private', 'types_count']
        
        categorical_cols = ['il', 'sehir_kategori', 'location_type']
        
        X = self.df[feature_columns].copy()
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                X[f'{col}_encoded'] = le.fit_transform(self.df[col].fillna('Unknown'))
        
        y = self.df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        feature_names = list(X.columns)
        importances = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("\nTop Important Features:")
        print(feature_importance.head(10))
        
        return self.model
    
    def setup_nearest_neighbors(self):
        print("Setting up nearest university recommender...")
        
        if self.df is None:
            raise ValueError("Data must be loaded first!")
        
        coords = self.df[['lat', 'lng']].values
        
        self.nn_model = NearestNeighbors(n_neighbors=4, metric='haversine', algorithm='ball_tree')
        self.nn_model.fit(np.radians(coords))
        
        self.university_data = self.df[['university_name', 'il', 'lat', 'lng', 'bolge']].copy()
        
        return self.nn_model
    
    def find_nearest_universities(self, lat, lng, n=3):
        if self.nn_model is None:
            raise ValueError("KNN model must be set up first!")
        
        query_point = np.radians([[lat, lng]])
        
        distances, indices = self.nn_model.kneighbors(query_point, n_neighbors=n+1)
        
        results = []
        for i, idx in enumerate(indices[0][1:]):
            uni = self.university_data.iloc[idx]
            distance_km = distances[0][i+1] * 6371
            
            results.append({
                'university': uni['university_name'],
                'il': uni['il'],
                'bolge': uni['bolge'],
                'distance_km': round(distance_km, 2),
                'lat': uni['lat'],
                'lng': uni['lng']
            })
        
        return results
    
    def create_map(self, center_lat=39.0, center_lng=35.0, zoom=6):
        print("Creating map...")
        
        if self.df is None:
            raise ValueError("Data must be loaded first!")
        
        m = folium.Map(
            location=[center_lat, center_lng], 
            zoom_start=zoom,
            tiles='OpenStreetMap'
        )
        
        region_colors = {
            'Marmara': 'blue',
            'Ege': 'green', 
            'Akdeniz': 'red',
            'İç Anadolu': 'orange',
            'Karadeniz': 'purple',
            'Doğu Anadolu': 'darkred',
            'Güneydoğu Anadolu': 'pink',
            'Diğer': 'gray'
        }
        
        marker_cluster = MarkerCluster().add_to(m)
        
        for idx, row in self.df.iterrows():
            color = region_colors.get(row['bolge'], 'gray')
            
            popup_text = f"""
            <b>{row['university_name']}</b><br>
            Location: {row['il']}, {row['ilce']}<br>
            Region: {row['bolge']}<br>
            Type: {'Private' if row['is_private'] else 'Public'}<br>
            Types: {row['types_count']}
            """
            
            folium.Marker(
                location=[row['lat'], row['lng']],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=row['university_name'][:30] + "...",
                icon=folium.Icon(color=color, icon='graduation-cap', prefix='fa')
            ).add_to(marker_cluster)
        
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 180px; height: 200px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Region Colors</b></p>
        '''
        
        for region, color in region_colors.items():
            if region != 'Diğer':
                legend_html += f'<p><i style="color:{color}">●</i> {region}</p>'
        
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def predict_location_category(self, lat, lng):
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained first!")
        
        il_encoded = 0
        sehir_kategori_encoded = 0
        location_type_encoded = 0
        
        if 40.5 < lat < 42.0 and 28.0 < lng < 30.0:
            il_encoded = 1
            sehir_kategori_encoded = 1
        elif 39.5 < lat < 40.5 and 32.0 < lng < 33.5:
            il_encoded = 2
            sehir_kategori_encoded = 1
        elif 38.0 < lat < 39.0 and 26.5 < lng < 28.5:
            il_encoded = 3
            sehir_kategori_encoded = 1
        
        features = np.array([[
            lat, lng, 1, 1, 1, 0, 3,
            il_encoded, sehir_kategori_encoded, location_type_encoded
        ]])
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        probability = max(self.model.predict_proba(features_scaled)[0])
        
        return prediction, round(probability, 3)
    
    def save_model(self, filename='university_model.pkl'):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'nn_model': self.nn_model,
            'university_data': self.university_data,
            'df': self.df
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved as {filename}!")
    
    def load_model(self, filename='university_model.pkl'):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.nn_model = model_data['nn_model']
        self.university_data = model_data['university_data']
        self.df = model_data['df']
        
        print(f"Model loaded from {filename}!")

def export_pickle_data(pickle_filename='turkiye_universities_model.pkl'):
    """Export pickle file contents to readable formats"""
    if not os.path.exists(pickle_filename):
        print(f"Pickle file {pickle_filename} not found!")
        return
    
    try:
        with open(pickle_filename, 'rb') as f:
            model_data = pickle.load(f)
        
        print("Pickle file contents:")
        print(f"- Keys: {list(model_data.keys())}")
        
        if 'df' in model_data:
            df = model_data['df']
            print(f"- DataFrame shape: {df.shape}")
            print(f"- Columns: {list(df.columns)}")
            
            # Export to CSV
            csv_filename = 'universities_data.csv'
            df.to_csv(csv_filename, index=False, encoding='utf-8')
            print(f"Data exported to {csv_filename}")
            
            # Export to JSON
            json_filename = 'universities_data.json'
            df.to_json(json_filename, orient='records', indent=2, force_ascii=False)
            print(f"Data exported to {json_filename}")
            
            # Show sample data
            print("\nSample data (first 5 rows):")
            print(df.head().to_string())
            
        if 'model' in model_data:
            model = model_data['model']
            print(f"- Model type: {type(model).__name__}")
            if hasattr(model, 'n_estimators'):
                print(f"- Number of estimators: {model.n_estimators}")
        
        print("Pickle export completed successfully!")
        
    except Exception as e:
        print(f"Error reading pickle file: {e}")

def main():
    project = UniversityMLProject()
    
    df = project.load_and_process_data('output.json')
    df = project.create_features_for_ml()
    model = project.train_ml_model(target_column='bolge')
    nn_model = project.setup_nearest_neighbors()
    
    map_obj = project.create_map()
    map_obj.save('turkiye_universities_map.html')
    print("Map saved as 'turkiye_universities_map.html'!")
    
    print("\nTest Examples:")
    
    ankara_lat, ankara_lng = 39.9334, 32.8597
    prediction, probability = project.predict_location_category(ankara_lat, ankara_lng)
    print(f"Ankara ({ankara_lat}, {ankara_lng})")
    print(f"   Prediction: {prediction} (Confidence: {probability*100:.1f}%)")
    
    nearest = project.find_nearest_universities(ankara_lat, ankara_lng, n=3)
    print("   Nearest Universities:")
    for i, uni in enumerate(nearest, 1):
        print(f"     {i}. {uni['university'][:40]}... - {uni['distance_km']} km")
    
    project.save_model('turkiye_universities_model.pkl')
    
    print("\nProject completed!")
    
    return project

def interactive_test(project):
    print("\nINTERACTIVE TEST MODE")
    print("=" * 50)
    
    while True:
        print("\nCoordinate Input:")
        print("(Type 'q' to quit)")
        
        lat_input = input("Latitude: ").strip()
        if lat_input.lower() == 'q':
            break
            
        lng_input = input("Longitude: ").strip()
        if lng_input.lower() == 'q':
            break
        
        try:
            lat = float(lat_input)
            lng = float(lng_input)
            
            print(f"\nAnalyzing: ({lat}, {lng})")
            print("-" * 30)
            
            prediction, probability = project.predict_location_category(lat, lng)
            print(f"ML Prediction: {prediction}")
            print(f"Confidence: {probability*100:.1f}%")
            
            print(f"\nNearest 3 Universities:")
            nearest = project.find_nearest_universities(lat, lng, n=3)
            
            for i, uni in enumerate(nearest, 1):
                print(f"  {i}. {uni['university'][:50]}...")
                print(f"     Location: {uni['il']} • Region: {uni['bolge']} • Distance: {uni['distance_km']} km")
            
            print(f"\nAnalysis completed!")
            
        except ValueError:
            print("Invalid coordinates! Please enter numbers.")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nInteractive test ended!")

if __name__ == "__main__":
    print("Turkey Universities ML Project Starting...")
    print("=" * 60)
    
    project = main()
    
    while True:
        print("\nWhat would you like to do?")
        print("1  Interactive Test (enter coordinates, get results)")
        print("2  Open Map (turkiye_universities_map.html)")
        print("3  Dataset Statistics")
        print("4  Export Pickle Data to CSV/JSON")
        print("5  Show Model Performance")
        print("q  Exit")
        
        choice = input("\nYour choice: ").strip()
        
        if choice == '1':
            interactive_test(project)
        
        elif choice == '2':
            if os.path.exists('turkiye_universities_map.html'):
                print("Map file 'turkiye_universities_map.html' created!")
                print("You can open it in your web browser.")
                
                try:
                    import webbrowser
                    webbrowser.open('turkiye_universities_map.html')
                    print("Map opened in browser!")
                except:
                    print("Please open 'turkiye_universities_map.html' manually.")
            else:
                print("Map file not found!")
        
        elif choice == '3':
            if project.df is not None:
                print(f"\nDATASET STATISTICS")
                print("-" * 40)
                print(f"Total Universities: {len(project.df)}")
                print(f"Public: {len(project.df[~project.df['is_private']])}")
                print(f"Private: {len(project.df[project.df['is_private']])}")
                print(f"Cities: {project.df['il'].nunique()}")
                
                print(f"\nREGION DISTRIBUTION:")
                bolge_counts = project.df['bolge'].value_counts()
                for bolge, count in bolge_counts.items():
                    print(f"   {bolge}: {count} universities")
                
                print(f"\nTOP CITIES BY UNIVERSITY COUNT:")
                sehir_counts = project.df['il'].value_counts().head(10)
                for sehir, count in sehir_counts.items():
                    print(f"   {sehir}: {count} universities")
        
        elif choice == '4':
            export_pickle_data('turkiye_universities_model.pkl')
        
        elif choice == '5':
            print(f"\nMODEL PERFORMANCE")
            print("-" * 30)
            print("Model successfully trained!")
            print("Can predict regions!")
            print("Can find nearest universities!")
            print("Interactive map created!")
        
        elif choice.lower() == 'q':
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice! Please enter 1-5 or 'q'.")