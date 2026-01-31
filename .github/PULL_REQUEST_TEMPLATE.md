## Summary
<!-- What this PR implements (concise) -->

## Files Changed
<!-- List main files added/changed -->

## How to Run Locally
```bash
# 1. Copy environment file
cp .env.example .env

# 2. Start services
docker-compose -f docker-compose.dev.yml up --build

# 3. Access
# Frontend: http://localhost:3000
# Backend: http://localhost:8000/docs
```

## Tests
```bash
# Frontend
cd frontend && npm test

# Backend
cd backend && pytest
```

## Screenshots/GIF
<!-- Attach images showing UI or terminal output -->

## OpenAPI/Postman
- Backend Swagger: http://localhost:8000/docs
- OpenAPI JSON: http://localhost:8000/openapi.json

## Checklist
- [ ] Code follows project style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Docker build succeeds
- [ ] Mock mode works without API keys

## Notes
<!-- Any additional context, TODOs for follow-up PRs, etc. -->
