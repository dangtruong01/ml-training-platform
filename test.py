from google.cloud import firestore
db = firestore.Client(project='ml-training-pipeline-sand-jjgq')
print('✅ Firestore connection successful')
print('Database:', db._database_string)