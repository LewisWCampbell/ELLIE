# Elle: realtime voice phone agent, one container
FROM node:22-bookworm-slim AS build
WORKDIR /app
# better-sqlite3 is a native module; needs a toolchain to compile.
RUN apt-get update && apt-get install -y --no-install-recommends python3 make g++ \
    && rm -rf /var/lib/apt/lists/*
COPY package*.json ./
RUN npm ci
COPY tsconfig.json ./
COPY src ./src
RUN npm run build

FROM node:22-bookworm-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production
COPY package*.json ./
RUN apt-get update && apt-get install -y --no-install-recommends python3 make g++ \
    && npm ci --omit=dev \
    && apt-get purge -y python3 make g++ && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/dist ./dist
COPY public ./public
# knowledge/ is the RAG corpus (resume, projects, skills) Elle answers from —
# it's read from the filesystem at runtime, so it must be in the image.
COPY knowledge ./knowledge
# profile.md is mounted or baked at deploy time for interview mode (optional).
EXPOSE 3000
CMD ["node", "dist/index.js"]
