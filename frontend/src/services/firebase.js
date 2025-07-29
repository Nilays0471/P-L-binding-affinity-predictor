// src/services/firebase.js
import { initializeApp } from "firebase/app";
import { 
  getAuth, 
  signInWithEmailAndPassword, 
  createUserWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithPopup,
  sendSignInLinkToEmail,
  isSignInWithEmailLink
} from "firebase/auth";

const firebaseConfig = {
    apiKey: "AIzaSyCVlEl7FkF9cvg03StE0aOHmcOB13WxkVE",
    authDomain: "pl-bindingaffinity.firebaseapp.com",
    projectId: "pl-bindingaffinity",
    storageBucket: "pl-bindingaffinity.firebasestorage.app",
    messagingSenderId: "618398553322",
    appId: "1:618398553322:web:a3a3fd73e72d4277e13393",
    measurementId: "G-NEJ1528YWD"
  };
const app  = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();
