# VORTA Ultra Frontend - React Dashboard
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Serve the application
EXPOSE 3000

CMD ["npm", "start"]
