#gcloud auth login 

gcloud config set project wrh-coord-platform  

gcloud run deploy stt-server `
  --source . `
  --platform managed `
  --region europe-west1 `
  --allow-unauthenticated `
  --env-vars-file .env-yaml `
  --memory=4Gi --cpu=2 --concurrency=1 `
  --min-instances=0