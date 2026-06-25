import os
import torch
import sys
import re
from collections import Counter
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import csv

# --- Mock torchvision for compatibility ---
class MockModule:
    def __getattr__(self, name): return MockModule()
    def __call__(self, *args, **kwargs): return MockModule()
sys.modules['torchvision'] = MockModule()
sys.modules['torchvision.ops'] = MockModule()
sys.modules['torchvision.transforms'] = MockModule()
if not hasattr(torch.ops, 'torchvision'):
    class DummyOps:
        def nms(*args, **kwargs): return torch.tensor([])
    torch.ops.torchvision = DummyOps()

# --------------------------------------------------
# Style Helpers
# --------------------------------------------------
def classify_style(name):
    if '_' in name:
        return 'snake_case'
    if any(c.isupper() for c in name) and name[0].islower():
        return 'camelCase'
    if name[0].isupper():
        return 'PascalCase'
    return 'lowercase' # Ambiguous

def get_style_outlier(text):
    """
    Returns the name of the identifier that violates the dominant style.
    """
    try:
        from tree_sitter_languages import get_parser
        parser = get_parser('java')
        tree = parser.parse(bytes(text, "utf8"))
        
        java_keywords = {"public", "static", "int", "void", "class", "for", "return"}
        names = []
        
        def traverse(node):
            if node.type == 'identifier':
                name = text[node.start_byte:node.end_byte]
                if name not in java_keywords:
                    names.append(name)
            for child in node.children:
                traverse(child)
        traverse(tree.root_node)
    except:
        # Fallback to regex
        names = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', text)
        names = [n for n in names if n not in {"int", "public", "static", "void"}]

    # Count styles of non-ambiguous names
    styles = []
    for n in set(names):
        style = classify_style(n)
        if style != 'lowercase':
            styles.append((n, style))
    
    if not styles: return None, None

    style_counts = Counter([s[1] for s in styles])
    dominant_style = style_counts.most_common(1)[0][0]
    
    outliers = [n for n, style in styles if style != dominant_style]
    
    if outliers:
        return outliers[0], dominant_style
    return None, dominant_style

# --------------------------------------------------
# Experiment Engine
# --------------------------------------------------
def run_style_experiment(tokenizer, model, masked_code, mask_token_id):
    print("\n" + "-"*50)
    print("Masked Code Snippet:")
    print(masked_code)
    
    # Identify dominant style from the remaining identifiers
    _, dominant_style = get_style_outlier(masked_code)
    
    if not dominant_style:
        print("Could not determine dominant style from context.")
        return None

    print(f"\nTarget Style (Dominant Style): {dominant_style}")

    inputs = tokenizer(masked_code, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")

    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=inputs.attention_mask.to("cuda"),
            max_length=input_ids.shape[1] + 16,
            steps=256,
            temperature=0, # Use Greedy for consistent style check
            return_dict_in_generate=True,
        )
    
    result_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    print(f"\nGenerated Code:\n{result_text}")
    
    # Simple extraction of what replaced the mask
    # We find the word at the position where <|mask|> used to be
    mask_start_idx = masked_code.find('<|mask|>')
    if mask_start_idx != -1:
        diff_match = re.search(r'\b[A-Za-z0-9_]+\b', result_text[mask_start_idx:])
        fixed_name = diff_match.group(0) if diff_match else "Unknown"
    else:
        fixed_name = "Unknown"

    print(f"\nDLLM Filled Name: '{fixed_name}'")
    fixed_style = classify_style(fixed_name)
    print(f"Filled Style: {fixed_style}")
    
    if fixed_style == dominant_style:
        print("✅ SUCCESS: DLLM followed the code style consistency!")
        success = True
    else:
        print("❌ FAILURE: DLLM did not follow the code style.")
        success = False
    
    return {
        "outlier_name": "PRE-MASKED",
        "dominant_style": dominant_style,
        "fixed_name": fixed_name,
        "fixed_style": fixed_style,
        "success": success
    }

def main():
    model_id = "apple/DiffuCoder-7B-Instruct"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    mask_token_id = tokenizer.convert_tokens_to_ids('<|mask|>')

    output_file = f"style_consistency_results_{datetime.now().strftime('%m%d_%H%M')}.csv"
    num_repeats = 50
    
    test_cases = [
        """
        public class matrix_utils {
            public void <|mask|><|mask|><|mask|>(int[][] matrix, int factor) {
                int rowCount = matrix.length;
                for (int i = 0; i < rowCount; i++) {
                    matrix[i][0] *= factor;
                }
            }
        }
        """,
        """
        public class Calculator {
            public int calculateSum(int <|mask|><|mask|><|mask|>, int secondNumber) {
                int totalSum = <|mask|><|mask|><|mask|> + secondNumber;
                int result_value = totalSum;
                return result_value;
            }
        }
        """,
        """
        public class data_processor {
            public void process_data(int <|mask|><|mask|><|mask|>) {
                int record_count = <|mask|><|mask|><|mask|>;
                printResults(record_count);
            }
            public void print_results(int count) {}
        }
        """,
        """
        public class StringHandler {
            public String formatText(String inputStr, int maxLength) {
                String resultStr = inputStr.trim();
                return <|mask|><|mask|><|mask|>(resultStr);
            }
            public String <|mask|><|mask|><|mask|>(String s) { return s; }
        }
        """,
        """
        public class list_manager {
            public void add_item(int new_item) {
                int list_size = 10;
                int currentPos = 0;
            }
            public void clear_list() {}
        }
        """,
        """
        public class AccountService {
            public void updateAccount(int accountId, double newBalance) {
                double oldBalance = 0.0;
                double TransactionAmount = newBalance - oldBalance;
            }
        }
        """,
        """
        public class File_Handler {
            public void write_file(String file_name) {
                String <|mask|><|mask|><|mask|> = "/tmp/" + file_name;
                saveToDisk(<|mask|><|mask|><|mask|>);
            }
            public void delete_file(String path) {}
        }
        """,
        """
        public class GeometryUtils {
            public double calculateCircleArea(double circleRadius) {
                double <|mask|><|mask|><|mask|> = 3.14159;
                double areaVal = <|mask|><|mask|><|mask|> * circleRadius * circleRadius;
                double circle_perimeter = 2 * <|mask|><|mask|><|mask|> * circleRadius;
                return areaVal;
            }
        }
        """,
        """
        public class user_logger {
            public void <|mask|><|mask|><|mask|>(String event_msg) {
                String timestamp_str = "10:00";
                System.out.println(timestamp_str + event_msg);
            }
            public void ClearLogs() {}
        }
        """,
        """
        public class OrderManager {
            public void processOrder(int orderId) {
                boolean isProcessed = true;
                <|mask|><|mask|><|mask|>(orderId);
            }
            public void archiveOrder(int id) {}
        }
        """,
        """
        public class sensor_data {
            public double <|mask|><|mask|><|mask|>() {
                double current_temp = 25.0;
                return current_temp;
            }
            public void setThreshold(double val) {}
        }
        """,
        """
        public class AuthHelper {
            private String <|mask|><|mask|><|mask|>;
            private int tokenExpiry;
            private String session_id;
            public void validateToken() {}
        }
        """,
        """
        public class database_client {
            private String <|mask|><|mask|><|mask|>;
            private int port_num;
            private String userName;
            public void connect_to_db() {}
        }
        """,
        """
        public class UIComponent {
            public void <|mask|><|mask|><|mask|>(int xPos, int yPos) {
                int widthVal = 100;
                int height_val = 50;
            }
        }
        """,
        """
        public class network_tool {
            public void ping_host(String <|mask|><|mask|><|mask|>) {
                int timeout_ms = 1000;
                int retryCount = 3;
            }
            public void tracert_host() {}
        }
        """,
        """
        public class SessionManager {
            public void <|mask|><|mask|><|mask|>(int userId) {
                String sessionId = "abc";
                long create_time = 12345L;
            }
        }
        """,
        """
        public class task_scheduler {
            public void <|mask|><|mask|><|mask|>(int task_id) {
                int delay_sec = 60;
                runTaskNow(task_id);
            }
            public void cancel_task(int id) {}
        }
        """,
        """
        public class InputValidator {
            public boolean isValidEmail(String <|mask|><|mask|><|mask|>) {
                String regexPattern = ".*";
                return <|mask|><|mask|><|mask|>.matches(regexPattern);
            }
            public boolean check_phone(String phone) { return true; }
        }
        """,
        """
        public class log_parser {
            public void <|mask|><|mask|><|mask|>(String log_line) {
                String[] parts_array = log_line.split(" ");
                process_parts(parts_array);
            }
            public void resetParser() {}
        }
        """,
        """
        public class BufferWrapper {
            public void <|mask|><|mask|><|mask|>(byte[] rawData) {
                int bufferSize = rawData.length;
                int offset_val = 0;
            }
        }
        """,
        """
        public class cache_service {
            public void put_entry(String key_name, Object val) {
                int <|mask|><|mask|><|mask|> = 3600;
                checkStatus(key_name);
            }
            public void clear_cache() {}
        }
        """,
        """
        public class RequestValidator {
            public void validateParams(Map<String, String> queryParams) {
                String apiKey = queryParams.get("key");
                String <|mask|><|mask|><|mask|> = queryParams.get("id");
            }
        }
        """,
        """
        public class db_transaction {
            public void begin_tx() {
                long start_time = System.currentTimeMillis();
                boolean isActive = true;
            }
            public void <|mask|><|mask|><|mask|>() {}
        }
        """,
        """
        public class ImageProcessor {
            public void <|mask|><|mask|><|mask|>(byte[] imageData) {
                int imgWidth = 800;
                int imgHeight = 600;
                process_pixels(imageData);
            }
            public void saveImage() {}
        }
        """,
        """
        public class worker_pool {
            private int <|mask|><|mask|><|mask|>;
            private int queue_size;
            private boolean is_running;
            public void startPool() {}
        }
        """,
        """
        public class NotificationManager {
            public void <|mask|><|mask|><|mask|>(String msgContent) {
                String targetId = "device1";
                send_msg(targetId, msgContent);
            }
            public void cancelNotification() {}
        }
        """,
        """
        public class config_loader {
            public void <|mask|><|mask|><|mask|>(String file_path) {
                String env_name = "dev";
                parseConfig(file_path);
            }
            public void save_defaults() {}
        }
        """,
        """
        public class UserSession {
            public void <|mask|><|mask|><|mask|>(String userName, String passWord) {
                boolean isAuth = true;
                set_token("token123");
            }
            public void logoutUser() {}
        }
        """,
        """
        public class api_client {
            public void make_request(String url_addr) {
                int <|mask|><|mask|><|mask|> = 3;
                handleResponse(url_addr);
            }
            public void set_timeout(int ms) {}
        }
        """,
        """
        public class EventDispatcher {
            public void dispatchEvent(Object eventObj) {
                String eventType = "CLICK";
                <|mask|><|mask|><|mask|>(eventObj);
            }
            public void registerListener() {}
        }
        """,
        """
        public class path_utils {
            public String <|mask|><|mask|><|mask|>(String file_name) {
                int dot_idx = file_name.lastIndexOf('.');
                return getSubString(file_name, dot_idx);
            }
            public boolean is_absolute(String path) { return false; }
        }
        """,
        """
        public class StreamHandler {
            public void readFromStream(InputStream <|mask|><|mask|><|mask|>) {
                int bytesRead = <|mask|><|mask|><|mask|>.read();
                processData(bytesRead);
            }
        }
        """,
        """
        public class query_builder {
            public String build_select(String <|mask|><|mask|><|mask|>) {
                String sql_query = "SELECT * FROM " + <|mask|><|mask|><|mask|>;
                return executeQuery(sql_query);
            }
        }
        """,
        """
        public class MathLibrary {
            public double calculateFactorial(int <|mask|><|mask|><|mask|>) {
                double resultVal = 1.0;
                for(int i_idx = 1; i_idx <= <|mask|><|mask|><|mask|>; i_idx++) {
                    resultVal *= i_idx;
                }
                return resultVal;
            }
        }
        """,
        """
        public class log_rotator {
            public void rotate_logs() {
                int <|mask|><|mask|><|mask|> = 5;
                long fileLimit = 1024L;
            }
            public void delete_old() {}
        }
        """,
        """
        public class StateMachine {
            public void transitionTo(String nextState) {
                String <|mask|><|mask|><|mask|> = "IDLE";
                update_ui(nextState);
            }
        }
        """,
        """
        public class disk_scanner {
            public void scan_directory(String dir_path) {
                int <|mask|><|mask|><|mask|> = 0;
                long totalSize = 0;
            }
        }
        """,
        """
        public class GeometryFactory {
            public Object createShape(String shapeType) {
                double <|mask|><|mask|><|mask|> = 1.0;
                return build_object(shapeType, <|mask|><|mask|><|mask|>);
            }
        }
        """,
        """
        public class socket_helper {
            public void open_connection(String ip_addr, int port_num) {
                boolean <|mask|><|mask|><|mask|> = true;
                send_handshake();
            }
        }
        """,
        """
        public class AppContext {
            public void initContext() {
                String app_name = "MyApp";
                String <|mask|><|mask|><|mask|> = "1.0";
            }
        }
        """
    ]

    with open(output_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["repeat_idx", "case_idx", "outlier_name", "dominant_style", "fixed_name", "fixed_style", "success"])
        writer.writeheader()

        for r in range(num_repeats):
            print(f"\n{'='*20} Starting Repeat {r+1}/{num_repeats} {'='*20}")
            for i, case in enumerate(test_cases):
                result = run_style_experiment(tokenizer, model, case.strip(), mask_token_id)
                if result:
                    result["repeat_idx"] = r
                    result["case_idx"] = i
                    writer.writerow(result)
                    f.flush() # Ensure data is written incrementally

    print(f"\nExperiment complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
