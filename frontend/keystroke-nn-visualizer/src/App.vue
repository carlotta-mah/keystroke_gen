<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import HelloWorld from './components/HelloWorld.vue'


import { ref } from 'vue';

// Define reactive variables
const participants = ref([]);
const selectedParticipant = ref(null);
const prediction = ref(null);

// Define methods
const trainModel = async () => {

};

const selectParticipant = async (participantId: number) => {
  // Fetch sequences for the selected participant from the backend
  // Update selectedParticipant data
};

const predict = async (sequenceId: number) => {
  // Send a request to the backend to make a prediction for the selected sequence
  // Update prediction data
};

</script>

<template>
  <header>
    <h1>Keystroke Model UI</h1>
  </header>
  <div class="controls">
    <button @click="trainModel">Train Model</button>
    <div v-if="participants.length">
    <h2>Participants</h2>
    <ul>
      <li v-for="participant in participants" :key="participant.id">
        <button @click="selectParticipant(participant.id)">{{ participant.name }}</button>
      </li>
    </ul>
  </div>
  </div>

<!--  <RouterView />-->
</template>

<style scoped>
header {
  line-height: 1.5;
  max-height: 100vh;
}

.controls {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}

.logo {
  display: block;
  margin: 0 auto 2rem;
}

nav {
  width: 100%;
  font-size: 12px;
  text-align: center;
  margin-top: 2rem;
}

nav a.router-link-exact-active {
  color: var(--color-text);
}

nav a.router-link-exact-active:hover {
  background-color: transparent;
}

nav a {
  display: inline-block;
  padding: 0 1rem;
  border-left: 1px solid var(--color-border);
}

nav a:first-of-type {
  border: 0;
}

@media (min-width: 1024px) {
  header {
    display: flex;
    place-items: center;
    padding-right: calc(var(--section-gap) / 2);
  }

  .logo {
    margin: 0 2rem 0 0;
  }

  header .wrapper {
    display: flex;
    place-items: flex-start;
    flex-wrap: wrap;
  }

  nav {
    text-align: left;
    margin-left: -1rem;
    font-size: 1rem;

    padding: 1rem 0;
    margin-top: 1rem;
  }
}
</style>
