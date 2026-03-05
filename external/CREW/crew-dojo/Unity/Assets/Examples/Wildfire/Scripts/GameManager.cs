using System;
using UnityEngine;
using Dojo;
using System.Collections.Generic;
using Nakama.TinyJson;
using Unity.Netcode;
using Dojo.Netcode;
using Unity.MLAgents;
using UnityEngine.AI;
using Unity.Netcode.Components;
using System.IO;

namespace Examples.Wildfire
{
    [DefaultExecutionOrder(-1)]
    public class GameManager : MonoBehaviour
    {
        [SerializeField]
        private DojoConnection _connection;
        private DojoTransport _transport;
        private AIAgentManager _agentManager;
        private MapManager mapManager;


        // frames before first fire can spawn
        private int startframes;
        private int frame;

        public int step;
        public int max_steps;
        public float tick_rate;


        public float ignition_chance;
        public float false_fire_chance;
        public bool fire_started;


        private bool IsClient => _connection.IsClient;
        public Camera serverCam;
        public RenderTexture serverTexture;
        public Camera clientCam;



        public AIAgent manager_agent;



        private static string starttime = $"{System.DateTime.Now:yyyy-MM-dd_HH-mm-ss}";

        public int frames_per_capture;
        public List<PlayerController> playercontrollers;


        private Vector2 spawn_location;
        public static List<int> returnVariables;

        public static int score;
        public static int base_score;

        private Texture2D _serverCaptureTex;
        private Dictionary<PlayerController, Texture2D> _minimapCaptureTexs = new();
        private Dictionary<PlayerController, Texture2D> _povCaptureTexs = new();
        private Texture2D _highResCache;
        private int _cacheWidth, _cacheHeight;



        private void Awake()
        {


            frames_per_capture = 5;
            ignition_chance = 1;
            false_fire_chance = 1;
            startframes = 0;
            score = 0;

            step = 0;
            fire_started = false;
            Application.targetFrameRate = 50;
            QualitySettings.vSyncCount = 0;
            Debug.Assert(FindObjectsOfType<GameManager>().Length == 1, "Only one game manager is allowed!");
            _connection = FindObjectOfType<DojoConnection>();
            _agentManager = GetComponentInChildren<AIAgentManager>();
            returnVariables = new List<int>();

            


            _connection.SubscribeRemoteMessages((long)NetOpCode.ClientAction, OnClientAction);
            _connection.SubscribeRemoteMessages((long)NetOpCode.ServerState, OnServerState);
            _connection.SubscribeRemoteMessages((long)NetOpCode.GameEvent, OnGameEvent);
            if (!_connection.IsClient)
            {
                serverCam.enabled = true;
                clientCam.enabled = true;
                serverCam.targetTexture = serverTexture;
            }
            else
            {
                serverCam.enabled = false;
                clientCam.enabled = true;
            }
        }
        private void Start()
        {
            mapManager = FindObjectOfType<MapManager>();
            NetworkManager.Singleton.OnServerStarted += OnServerStarted;
            var args = Environment.GetCommandLineArgs();

            for (var idx = 0; idx < args.Length; ++idx)
            {
                var arg = args[idx];

                if (arg.Equals("-DecisionRequestFrequency") && idx < args.Length - 1 && float.TryParse(args[idx + 1], out var requestFreq))
                {
                    tick_rate = requestFreq;
                    ++idx;
                }
                tick_rate = 2f;

                if (arg.Equals("-MaxSteps") && idx < args.Length - 1 && int.TryParse(args[idx + 1], out var maxStep))
                {
                    max_steps = maxStep;
                    ++idx;
                }
                if (arg.Equals("-ServerCamSize") && idx < args.Length - 1 && int.TryParse(args[idx + 1], out var camSize))
                {
                    serverCam.orthographicSize = 0.5f / camSize;
                    ++idx;
                }
                max_steps = 600;

            }
            


        }
        public void SetUpGame()
        {
            returnVariables.Add((int)ConfigReader.game_type);
            returnVariables.Add(ConfigReader.map_size);
            returnVariables.Add(score);

            System.Random random = new System.Random(ConfigReader.seed);

            int spawn_x = random.Next((int)(ConfigReader.map_size * 0.8f)) + (int)(ConfigReader.map_size * 0.1f);
            int spawn_y = random.Next((int)(ConfigReader.map_size * 0.8f)) + (int)(ConfigReader.map_size * 0.1f);
            Debug.Log("spawn location: " + spawn_x + ", " + spawn_y);
            spawn_location = new Vector2(spawn_x, spawn_y);

            if (ConfigReader.game_type == GameType.CutTrees)
            {
                returnVariables.Add(ConfigReader.lines ? (1) : (0));
                returnVariables.Add(ConfigReader.tree_count);
                Debug.Log("tree count: " + ConfigReader.tree_count);
                Debug.Log("map size: " + ConfigReader.map_size);

                HashSet<(int, int)> chosen = new HashSet<(int, int)>();
                if (!ConfigReader.lines)
                {

                    for (int i = 0; i < ConfigReader.tree_count; i++)
                    {
                        bool valid = false;
                        while (!valid)
                        {
                            int tree_x = random.Next((int)(ConfigReader.map_size * 0.8f)) + (int)(ConfigReader.map_size * 0.1f);
                            int tree_y = random.Next((int)(ConfigReader.map_size * 0.8f)) + (int)(ConfigReader.map_size * 0.1f);

                            Debug.Log(tree_x + ", " + tree_y);

                            Cell currCell = mapManager.cellGrid.grid[(int)tree_y][(int)tree_x];


                            float min_distance = 0.15f * ConfigReader.map_size;
                            Debug.Log("mindistance: " + min_distance);
                            if (Math.Abs(spawn_location.x - tree_x) > min_distance && Math.Abs(spawn_location.y - tree_y) > min_distance && mapManager.cellGrid.grid[(int)tree_y][(int)tree_x].trees != null && !chosen.Contains((tree_x, tree_y)))
                            {
                                Debug.Log("valid");
                                valid = true;
                                returnVariables.Add(tree_x);
                                returnVariables.Add(tree_y);
                                chosen.Add((tree_x, tree_y));
                                base_score += 3-mapManager.cellGrid.grid[(int)tree_y][(int)tree_x].trees.count;
                            }
                            else
                            {
                                Debug.Log("invalid");
                            }

                        }
                    }

                }
                else
                {
                    for (int i = 0; i < ConfigReader.tree_count; i++)
                    {
                        bool valid = false;
                        while (!valid)
                        {
                            int tree_x = random.Next((int)(ConfigReader.map_size * 0.8f)) + (int)(ConfigReader.map_size * 0.1f);
                            int tree_y = random.Next((int)(ConfigReader.map_size * 0.8f)) + (int)(ConfigReader.map_size * 0.1f);

                            Debug.Log(tree_x + ", " + tree_y);


                            float min_distance = 0.2f * ConfigReader.map_size;
                            Debug.Log("mindistance: " + min_distance);
                            if (Math.Abs(spawn_location.x - tree_x) > min_distance && Math.Abs(spawn_location.y - tree_y) > min_distance&& !chosen.Contains((tree_x, tree_y)))
                            {
                                Debug.Log("valid");
                                valid = true;

                                bool isHorizontal = random.Next(2) == 0; // 50% chance for horizontal or vertical

                                int halfLength = ConfigReader.trees_per_line / 2;
                                int startX, startY, endX, endY;

                                if (isHorizontal)
                                {
                                    // Adjust centerX to ensure the line fits within the grid
                                    tree_x = Math.Max(halfLength, Math.Min(tree_x, ConfigReader.map_size - halfLength - 1));
                                    startX = tree_x - halfLength;
                                    endX = tree_x + (ConfigReader.trees_per_line-halfLength);
                                    startY = endY = tree_y; // Same Y-coordinate
                                }
                                else
                                {
                                    // Adjust centerY to ensure the line fits within the grid
                                    tree_y = Math.Max(halfLength, Math.Min(tree_y, ConfigReader.map_size - halfLength - 1));
                                    startY = tree_y - halfLength;
                                    endY = tree_y + (ConfigReader.trees_per_line - halfLength);
                                    startX = endX = tree_x; // Same X-coordinate
                                }

                                chosen.Add((tree_x, tree_y));
                                returnVariables.Add(startX);
                                returnVariables.Add(startY);
                                returnVariables.Add(endX);
                                returnVariables.Add(endY);



                            }
                            else
                            {
                                Debug.Log("invalid");
                            }

                        }
                    }

                }
            }


            else if (ConfigReader.game_type == GameType.ScoutFire)
            {
                bool fvalid = false;
                while (!fvalid)
                {
                    int fire_x = random.Next((int)(ConfigReader.map_size * 0.9f)) + (int)(ConfigReader.map_size * 0.05f);
                    int fire_y = random.Next((int)(ConfigReader.map_size * 0.9f)) + (int)(ConfigReader.map_size * 0.05f);

                    Debug.Log(fire_x + ", " + fire_y);


                    float min_distance = 0.4f * ConfigReader.map_size;
                    Debug.Log("mindistance: " + min_distance);
                    if (Math.Abs(spawn_location.x - fire_x) > min_distance || Math.Abs(spawn_location.y - fire_y) > min_distance)
                    {
                        fvalid = mapManager.SetFire((float)fire_x, (float)fire_y);
                        Debug.Log("fire start " + fvalid);

                        if (fvalid)
                        {
                            returnVariables.Add(fire_x);
                            returnVariables.Add(fire_y);

                        }


                    }
                    else
                    {
                        Debug.Log("invalid");
                    }

                }
            }

            else if (ConfigReader.game_type == GameType.PickAndPlace)
            {
                bool pvalid = false;
                while (!pvalid)
                {
                    int target_x = random.Next((int)(ConfigReader.map_size * 0.9f)) + (int)(ConfigReader.map_size * 0.05f);
                    int target_y = random.Next((int)(ConfigReader.map_size * 0.9f)) + (int)(ConfigReader.map_size * 0.05f);

                    Debug.Log(target_x + ", " + target_y);
                    float min_distance = 0.4f * ConfigReader.map_size;
                    Debug.Log("mindistance: " + min_distance);
                    if (Math.Abs(spawn_location.x - target_x) > min_distance && Math.Abs(spawn_location.y - target_y) > min_distance)
                    {
                        returnVariables.Add(target_x);
                        returnVariables.Add(target_y);
                        pvalid = true;

                    }
                }
            }
            
            else if (ConfigReader.game_type == GameType.ContainFire)
            {
                Debug.Log("water: " + ConfigReader.water.ToString());
                bool fvalid = false;
                while (!fvalid)
                {
                    int fire_x = random.Next((int)(ConfigReader.map_size * 0.8f)) + (int)(ConfigReader.map_size * 0.1f);
                    int fire_y = random.Next((int)(ConfigReader.map_size * 0.8f)) + (int)(ConfigReader.map_size * 0.1f);

                    Debug.Log("fire location" + fire_x + ", " + fire_y);

                    float min_distance = 0f * ConfigReader.map_size;
                    //float min_distance = 0.3f * ConfigReader.map_size;
                    Debug.Log("mindistance: " + min_distance);
                    if (Math.Abs(spawn_location.x - fire_x) > min_distance || Math.Abs(spawn_location.y - fire_y) > min_distance)
                    {
                        fvalid = mapManager.SetFire((float)fire_x, (float)fire_y);
                        Debug.Log("fire start " + fvalid);

                        if (fvalid)
                        {
                            returnVariables.Add(fire_x);
                            returnVariables.Add(fire_y);


                            if (ConfigReader.water)
                            {
                                bool wvalid = false;
                                while (!wvalid)
                                {
                                    int water_x = random.Next((int)(ConfigReader.map_size * 0.9f)) + (int)(ConfigReader.map_size * 0.05f);
                                    int water_y = random.Next((int)(ConfigReader.map_size * 0.9f)) + (int)(ConfigReader.map_size * 0.05f);


                                    if (water_x != fire_x || water_y != fire_y)
                                    {
                                        Debug.Log("water location" + water_x + ", " + water_y);
                                        mapManager.setWater(new Vector2(water_x, water_y));
                                        returnVariables.Add(water_x);
                                        returnVariables.Add(water_y);
                                        wvalid = true;

                                    }
                                }

                                
                            }

                        }
                    }
                    else
                    {
                        Debug.Log("invalid");
                    }

                }


            }

            else if (ConfigReader.game_type == GameType.MoveCivilians)
            {

                int target_x = random.Next((int)(ConfigReader.map_size * 0.8f)) + (int)(ConfigReader.map_size * 0.1f);
                int target_y = random.Next((int)(ConfigReader.map_size * 0.8f)) + (int)(ConfigReader.map_size * 0.1f);
                returnVariables.Add(target_x);
                returnVariables.Add(target_y);
                Debug.Log("target: "+ target_x + ", " + target_y);

                for (int i = 0; i < ConfigReader.civilian_cluster_count; i++)
                {
                    bool valid = false;
                    float min_distance = 0.3f * ConfigReader.map_size;
                    float min_player_distance = 0.2f * ConfigReader.map_size;

                    while (!valid)
                    {
                        Vector2 civ_spawn_location = new Vector2(random.Next((int)(ConfigReader.map_size * 0.8f)) + (int)(ConfigReader.map_size * 0.1f), random.Next((int)(ConfigReader.map_size * 0.8f)) + (int)(ConfigReader.map_size * 0.1f));
                        Debug.Log("attempt: " + civ_spawn_location.x + ", " + civ_spawn_location.y);

                        if ((Math.Abs(civ_spawn_location.x - target_x) > min_distance || Math.Abs(civ_spawn_location.y - target_y) > min_distance) && (Math.Abs(civ_spawn_location.x - spawn_x) > min_player_distance || Math.Abs(civ_spawn_location.y - spawn_y) > min_player_distance))
                        {
                            mapManager.SpawnCivilians(ConfigReader.civilian_count, civ_spawn_location);
                            returnVariables.Add((int)civ_spawn_location.x);
                            returnVariables.Add((int)civ_spawn_location.y);
                            Debug.Log("civ_spawn: " + civ_spawn_location.x + ", " + civ_spawn_location.y);
                            valid = true;
                        }
                    }
     

                }


            }
            else if (ConfigReader.game_type== GameType.Both)
            {

                bool fvalid = false;
                while (!fvalid)
                {
                    int fire_x = random.Next((int)(ConfigReader.map_size * 0.7f)) + (int)(ConfigReader.map_size * 0.15f);
                    int fire_y = random.Next((int)(ConfigReader.map_size * 0.7f)) + (int)(ConfigReader.map_size * 0.15f);

                    Debug.Log("fire location" + fire_x + ", " + fire_y);


                    float min_distance = 0.25f * ConfigReader.map_size;
                    Debug.Log("mindistance: " + min_distance);
                    if (Math.Abs(spawn_location.x - fire_x) > min_distance || Math.Abs(spawn_location.y - fire_y) > min_distance)
                    {
                        fvalid = mapManager.SetFire((float)fire_x, (float)fire_y);
                        Debug.Log("fire start " + fvalid);

                        if (fvalid)
                        {
                            returnVariables.Add(fire_x);
                            returnVariables.Add(fire_y);


                            if (ConfigReader.water)
                            {
                                bool wvalid = false;
                                while (!wvalid)
                                {
                                    int water_x = random.Next((int)(ConfigReader.map_size * 0.9f)) + (int)(ConfigReader.map_size * 0.05f);
                                    int water_y = random.Next((int)(ConfigReader.map_size * 0.9f)) + (int)(ConfigReader.map_size * 0.05f);

                                    if (water_x != fire_x && water_y != fire_y)
                                    {
                                        mapManager.setWater(new Vector2(water_x, water_y));
                                        returnVariables.Add(water_x);
                                        returnVariables.Add(water_y);
                                        wvalid = true;

                                    }
                                }
                            }
                            else{
                                returnVariables.Add(0);
                                returnVariables.Add(0);
                            }
                            for (int i = 0; i < ConfigReader.civilian_cluster_count; i++)
                            {
                                bool cvalid = false;
                                while (!cvalid)
                                {
                                    int civ_spawn_x = random.Next((int)(ConfigReader.map_size * 0.8f)) + (int)(ConfigReader.map_size * 0.1f);
                                    int civ_spawn_y = random.Next((int)(ConfigReader.map_size * 0.8f)) + (int)(ConfigReader.map_size * 0.1f);

                                    if (Math.Abs(civ_spawn_x - fire_x) > min_distance && Math.Abs(civ_spawn_y - fire_y) > min_distance)
                                    {
                                        Vector2 civ_spawn_location = new Vector2(civ_spawn_x, civ_spawn_y);
                                        Debug.Log("spawning civilians");
                                        mapManager.SpawnCivilians(ConfigReader.civilian_count, civ_spawn_location);
                                        returnVariables.Add((int)civ_spawn_location.x);
                                        returnVariables.Add((int)civ_spawn_location.y);
                                        cvalid = true;

                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        Debug.Log("invalid");
                    }

                }

                

            }
            else
            {
                Debug.Log("invalid game type");
            }

            Debug.Log("game set up");
        }


        private void OnServerStarted()
        {
            if (NetworkManager.Singleton.IsServer)
            {
                _transport = NetworkManager.Singleton.NetworkConfig.NetworkTransport as DojoTransport;
                Debug.Log("spawning agents");
                _agentManager.SpawnAgents(spawn_location);
                this.manager_agent = _agentManager.Agents[0];
                //mapManager.SpawnCivilians(20,4);

            }

        }
        private void OnClientAction(DojoMessage m)
        {
            if (!IsClient)
            {
                var action = m.GetString();
                if (Enum.TryParse(typeof(NetCommand), action, out var command))
                {
                    //_board.HandleClientControl((NetCommand)command);
                }
                else
                {
                    Debug.LogWarning($"Invalid remote action: {action}");
                }
            }
        }


        public void SaveHighResTexture(Texture2D original, int upscaleFactor, string savePath)
        {
            int newWidth = original.width * upscaleFactor;
            int newHeight = original.height * upscaleFactor;

            // if our cache is null or wrong size, rebuild it
            if (_highResCache == null || newWidth != _cacheWidth || newHeight != _cacheHeight)
            {
                if (_highResCache != null)
                    Destroy(_highResCache);

                _highResCache = new Texture2D(newWidth, newHeight, TextureFormat.RGBA32, false);
                _cacheWidth = newWidth;
                _cacheHeight = newHeight;
            }

            // copy pixels
            for (int y = 0; y < original.height; y++)
            {
                for (int x = 0; x < original.width; x++)
                {
                    Color c = original.GetPixel(x, y);
                    int baseX = x * upscaleFactor;
                    int baseY = (original.height - 1 - y) * upscaleFactor; // Flip Y coordinate
                    for (int dy = 0; dy < upscaleFactor; dy++)
                        for (int dx = 0; dx < upscaleFactor; dx++)
                            _highResCache.SetPixel(baseX + dx, baseY + dy, c);
                }
            }
            _highResCache.Apply();

            // encode & write
            byte[] bytes = _highResCache.EncodeToPNG();
            File.WriteAllBytes(savePath, bytes);
        }
        public void FixedUpdate()
        {
            if (!_connection.IsClient)
            {
                frame += 1;
                step = (int)(frame / (50 * tick_rate));

                if (step == max_steps && !_connection.IsClient)
                {
                    Debug.Log("Max Steps Reached");
                    EndGame();
                }


                //if (!fire_started)
                //{
                //    if(UnityEngine.Random.value < false_fire_chance)
                //    {
                //        mapManager.FalseFire(UnityEngine.Random.value, UnityEngine.Random.value);
                //    }


                //    if (startframes < 0 && UnityEngine.Random.value < ignition_chance)
                //    {
                //        fire_started = mapManager.SetRandomFire(UnityEngine.Random.value, UnityEngine.Random.value);
                //    }
                //    startframes -= 1;
                //}
                //else
                //{
                //    if (!mapManager.cellGrid.hasIgnitedCell)
                //    {
                //        Debug.Log("Fire Extinguished");
                //        EndGame();
                //    }
                //}


                
                playercontrollers =  new List<PlayerController>();
                foreach(PlayerController p in mapManager.firefighters)
                {
                    playercontrollers.Add(p);
                }
                foreach (PlayerController p in mapManager.bulldozers)
                {
                    playercontrollers.Add(p);
                }
                foreach (PlayerController p in mapManager.drones)
                {
                    playercontrollers.Add(p);
                }
                foreach (PlayerController p in mapManager.helicopters)
                {
                    playercontrollers.Add(p);
                }

                if (playercontrollers.Count > 0) {

                    ComputeCoveringSquare(playercontrollers, 30, out Vector3 center, out double squareSize);


                    serverCam.transform.position = center-new Vector3(0,100,0);
                    serverCam.orthographicSize = (0.5f * (float)squareSize) / MapManager.mapSize.x;
                    
                    if (((frame % frames_per_capture) ==0)&& ConfigReader.log_trajectory)
                    {
                        SaveRenders(frame / frames_per_capture);
                    }
                    CheckScore();


                }

            }
        }

        public static void ComputeCoveringSquare(List<PlayerController> points, double flatMargin, out Vector3 center, out double squareSize)
        {


            // Initialize min and max with the first point.
            double minX = points[0].transform.position.x;
            double maxX = points[0].transform.position.x;
            double minY = points[0].transform.position.z;
            double maxY = points[0].transform.position.z;

            // Find the bounding box.
            foreach (var p in points)
            {
                if (p.transform.position.x < minX) minX = p.transform.position.x;
                if (p.transform.position.x > maxX) maxX = p.transform.position.x;
                if (p.transform.position.y < minY) minY = p.transform.position.z;
                if (p.transform.position.y > maxY) maxY = p.transform.position.z;
            }

            // Calculate the midpoint of the bounding box.
            center = new Vector3((float)(minX + maxX) / (2f * MapManager.mapSize.x), (float)(minY + maxY) / (2f * MapManager.mapSize.y), -0.2f);

            // Determine the minimal required square size (the larger dimension of the bounding box).
            double width = maxX - minX;
            double height = maxY - minY;
            double minimalSquareSize = Math.Max(width, height);

            // Apply the flat margin: add margin on both sides.
            squareSize = minimalSquareSize + 2 * flatMargin;
        }

        public void SaveRenders(int step)
        {
            string folder = Path.Combine(
                            ConfigReader.render_folder_path,
                            $"wildfire_alg/results/logs/{ConfigReader.algorithm}/{ConfigReader.level}/{ConfigReader.seed}/{ConfigReader.timestamp}/Server_Map"
                        );
            if (!Directory.Exists(folder))
                Directory.CreateDirectory(folder);

            string filename = Path.Combine(folder, $"capture_{step}.png");

            // now this will reuse one big Texture2D and won't leak:
            SaveHighResTexture(mapManager.textureMap, 32, filename);
            // --- 1) Server accumulative capture ---
            if (_serverCaptureTex == null)
                _serverCaptureTex = new Texture2D(serverTexture.width, serverTexture.height, TextureFormat.RGB24, false);

            string baseFolder = Path.Combine(
                ConfigReader.render_folder_path,
                $"wildfire_alg/results/logs/{ConfigReader.algorithm}/{ConfigReader.level}/{ConfigReader.seed}/{ConfigReader.timestamp}"
            );
            Debug.Log(baseFolder);
            CaptureTextureToFile(
                serverTexture,
                _serverCaptureTex,
                Path.Combine(baseFolder, "Server_Accumulative"),
                step
            );

            // --- 2) Per-agent captures ---
            foreach (var p in playercontrollers)
            {
                // Minimap
                if (!_minimapCaptureTexs.TryGetValue(p, out var miniTex))
                {
                    miniTex = new Texture2D(p.minimapTexture.width, p.minimapTexture.height, TextureFormat.RGB24, false);
                    _minimapCaptureTexs[p] = miniTex;
                }
                CaptureTextureToFile(
                    p.minimapTexture,
                    miniTex,
                    Path.Combine(baseFolder, $"Agent_{p.agent.AgentId}/Minimap"),
                    step
                );

                // POV
                if (!_povCaptureTexs.TryGetValue(p, out var povTex))
                {
                    povTex = new Texture2D(p.povTexture.width, p.povTexture.height, TextureFormat.RGB24, false);
                    _povCaptureTexs[p] = povTex;
                }
                CaptureTextureToFile(
                    p.povTexture,
                    povTex,
                    Path.Combine(baseFolder, $"Agent_{p.agent.AgentId}/POV"),
                    step
                );
            }
        }


        private void CaptureTextureToFile(RenderTexture rt, Texture2D tex, string folderPath, int step)
        {

            RenderTexture.active = rt;
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply();
            RenderTexture.active = null;


            byte[] bytes = tex.EncodeToPNG();
            if (!Directory.Exists(folderPath))
                Directory.CreateDirectory(folderPath);

            string filename = Path.Combine(folderPath, $"capture_{step}.png");
            File.WriteAllBytes(filename, bytes);
        }
        private void OnDestroy()
        {
            // free GPU memory
            if (_serverCaptureTex != null) Destroy(_serverCaptureTex);
            foreach (var kv in _minimapCaptureTexs) Destroy(kv.Value);
            foreach (var kv in _povCaptureTexs) Destroy(kv.Value);
        }

        public void CheckScore()
        {
            int temp_score = 0;
            if (ConfigReader.game_type == GameType.CutTrees)
            {
                if (!ConfigReader.lines)
                {

                    for (int i = 5; i < returnVariables.Count; i += 2)
                    {
                        int tree_x = returnVariables[i];
                        int tree_y = returnVariables[i + 1];

                        Cell currCell = mapManager.cellGrid.grid[(int)tree_y][(int)tree_x];

                        if (currCell.trees != null)
                        {
                            temp_score += (3- currCell.trees.count);
                        }
                        else
                        {
                            temp_score += 3;
                        }
                    }
                }
                else
                {
                    for (int i = 5; i < returnVariables.Count; i += 4)
                    {
                        int start_x = returnVariables[i];
                        int start_y = returnVariables[i + 1];
                        int end_x = returnVariables[i+2];
                        int end_y = returnVariables[i + 3];



                        if (start_x == end_x) // Vertical line
                        {
                            int minY = Math.Min(start_y, end_y);
                            int maxY = Math.Max(start_y, end_y);
                            for (int y = minY; y <= maxY; y++)
                            {
                                Cell c = mapManager.cellGrid.grid[(int)y][(int)start_x];
                                if (c.trees == null)
                                {
                                    temp_score += 3;
                                    //Debug.Log($"null tree at {start_x}, {y}");
                                }
                                else
                                {
                                    temp_score += (3-c.trees.count);
                                    //Debug.Log($"{c.trees.count} trees at {start_x}, {y}");
                                }

                            }
                        }
                        else if (start_y == end_y) // Horizontal line
                        {
                            int minX = Math.Min(start_x, end_x);
                            int maxX = Math.Max(start_x, end_x);
                            for (int x = minX; x <= maxX; x++)
                            {
                                Cell c = mapManager.cellGrid.grid[(int)start_y][(int)x];
                                if (c.trees == null)
                                {
                                    temp_score += 3;
                                    //Debug.Log($"null tree at {x}, {start_y}");
                                }
                                else
                                {
                                    temp_score += (3 - c.trees.count);
                                    //Debug.Log($"{c.trees.count} trees at {x}, {start_y}");
                                }
                            }
                        }


                    }
                }
            }
            else if (ConfigReader.game_type == GameType.ScoutFire)
            {
                int scouted = 0;

                foreach (Drone d in mapManager.drones)
                {
                    CellState currCellState = mapManager.cellGrid.grid[(int)d.gridPos.y][(int)d.gridPos.x].state;
                    if (currCellState == CellState.on_fire || currCellState == CellState.extenguishing || currCellState == CellState.fully_extenguished)
                    {
                        scouted += 1;
                    }
                }
                temp_score = Math.Min(scouted, 2);
            }
            else if (ConfigReader.game_type == GameType.PickAndPlace)
            {
                int at_target = 0;

                int target_x = returnVariables[3];
                int target_y = returnVariables[4];
                foreach (Firefighter f in mapManager.firefighters)
                {
                    if ((Math.Abs(target_x - f.gridPos.x) <= 3) && (Math.Abs(target_y - f.gridPos.y) <= 3))
                    {
                        at_target += 1;
                    }
                }
                temp_score = at_target;
            }
            else if (ConfigReader.game_type == GameType.ContainFire)
            {
                // Trees (1), Buildings (100), Civilians (1000), Firefighters (1000), Bulldozers (1500)
                temp_score = -(MapManager.tree_destroyed + (MapManager.buildings_destroyed * 10) + (MapManager.civilians_destroyed * 100) + (MapManager.firefighters_destroyed *20) + (MapManager.bulldozers_destroyed * 20));
            }
            else if (ConfigReader.game_type == GameType.MoveCivilians)
            {
                int at_target = 0;

                int target_x = returnVariables[3];
                int target_y = returnVariables[4];



                foreach (Civilian c in mapManager.civilians)
                {
                    if ((Math.Abs(target_x - (int)c.gridPos.x) <= 3) && (Math.Abs(target_y - (int)c.gridPos.y) <= 3))
                    {
                        at_target += 1;
                        c.active = false;
                    }
                }
                temp_score = at_target;
            }
            else if(ConfigReader.game_type == GameType.Both)
            {
                temp_score = -(MapManager.tree_destroyed + (MapManager.buildings_destroyed * 10) + (MapManager.civilians_destroyed * 100) + (MapManager.firefighters_destroyed * 20) + (MapManager.bulldozers_destroyed * 20));

            }


            //temp_score -= base_score;
            returnVariables[2] = temp_score;
        }

        private void EndGame()
        {

        }

        private void OnServerState(DojoMessage m)
        {
        }
        private void OnGameEvent(DojoMessage m)
        {

        }

        
        

    }
}

