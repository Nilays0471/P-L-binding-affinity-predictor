import { auth, googleProvider } from "../services/firebase";
import { signInWithEmailAndPassword, signInWithPopup } from "firebase/auth";

function LoginForm() {
  const [email, setEmail] = useState(""), [password, setPassword] = useState("");
  const handleEmailLogin = (e) => {
    e.preventDefault();
    signInWithEmailAndPassword(auth, email, password)
      .then(userCred => "/dashboard")
      .catch(err => err.message);
  };
  const handleGoogle = () => {
    signInWithPopup(auth, googleProvider)
      .then(cred => "/dashboard")
      .catch(console.error);
  };
  return (
    <form onSubmit={handleEmailLogin}>
      <input value={email} onChange={e=>setEmail(e.target.value)} placeholder="Email"/>
      <input type="password" value={password} onChange={e=>setPassword(e.target.value)} placeholder="Password"/>
      <button type="submit">Login</button>
      <button type="button" onClick={handleGoogle}>Login with Google</button>
    </form>
  );
}
