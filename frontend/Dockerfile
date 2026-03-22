FROM node:18-alpine AS build

WORKDIR /app

ARG VITE_MODEL_ENDPOINT=/infer
ENV VITE_MODEL_ENDPOINT=${VITE_MODEL_ENDPOINT}

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build:production

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
