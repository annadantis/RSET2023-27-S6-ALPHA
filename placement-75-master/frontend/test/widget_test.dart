import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:placement_assistant/providers/auth_provider.dart';
import 'package:placement_assistant/screens/login_screen.dart';

void main() {
  testWidgets('Login screen renders correctly test', (WidgetTester tester) async {
    // 1. Build our app within the Provider scope.
    // We wrap it in a ChangeNotifierProvider because LoginScreen 
    // depends on AuthProvider to function.
    await tester.pumpWidget(
      MultiProvider(
        providers: [
          ChangeNotifierProvider(create: (_) => AuthProvider()),
        ],
        child: const MaterialApp(
          home: LoginScreen(),
        ),
      ),
    );

    // 2. Verify that the Login Screen elements are present.
    expect(find.text('AI Placement Assistant'), findsOneWidget);
    expect(find.byType(TextField), findsNWidgets(2)); // Username and Password fields
    expect(find.text('Login'), findsOneWidget);
    expect(find.text('New User? Register here'), findsOneWidget);

    // 3. Test interaction: Enter text into the username field.
    await tester.enterText(find.byType(TextField).first, 'testuser');
    expect(find.text('testuser'), findsOneWidget);
  });
}
