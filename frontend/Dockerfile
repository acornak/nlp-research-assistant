FROM node:22.8.0-alpine AS build

WORKDIR /app

COPY package.json yarn.lock ./

RUN yarn install

ENV PATH /app/node_modules/.bin:$PATH

COPY . .

ARG VITE_API_URL
ARG VITE_MLFLOW_URL

RUN echo "VITE_API_URL=${VITE_API_URL}" > .env && \
    echo "VITE_MLFLOW_URL=${VITE_MLFLOW_URL}" >> .env

RUN yarn build

FROM nginx:1.27.1-alpine

COPY ./nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /var/www/html/

EXPOSE 80

ENTRYPOINT ["nginx","-g","daemon off;"]
